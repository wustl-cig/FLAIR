import cv2
import numpy as np
import os
import torch
from torchvision.transforms.functional import normalize
from einops import rearrange
from guided_diffusion.facelib.detection import init_detection_model
from guided_diffusion.facelib.parsing import init_parsing_model
from guided_diffusion.facelib.utils.misc import (
    img2tensor,
    imwrite,
    is_gray,
    bgr2gray,
    adain_npy,
)
from guided_diffusion.facelib.utils.misc import load_file_from_url, get_device
from torchvision.transforms import functional as VF
from guided_diffusion.facelib.detection.retinaface.retinaface import RetinaFace
from guided_diffusion.facelib.parsing.parsenet import ParseNet

dlib_model_url = {
    "face_detector": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/mmod_human_face_detector-4cb19393.dat",
    "shape_predictor_5": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/shape_predictor_5_face_landmarks-c4b1e980.dat",
}


def get_largest_face(det_faces, h, w):
    def get_location(val, length):
        if val < 0:
            return 0
        elif val > length:
            return length
        else:
            return val

    face_areas = []
    for det_face in det_faces:
        left = get_location(det_face[0], w)
        right = get_location(det_face[2], w)
        top = get_location(det_face[1], h)
        bottom = get_location(det_face[3], h)
        face_area = (right - left) * (bottom - top)
        face_areas.append(face_area)
    largest_idx = face_areas.index(max(face_areas))
    return det_faces[largest_idx], largest_idx


def get_center_face(det_faces, h=0, w=0, center=None):
    if center is not None:
        center = np.array(center)
    else:
        center = np.array([w / 2, h / 2])
    center_dist = []
    for det_face in det_faces:
        face_center = np.array(
            [(det_face[0] + det_face[2]) / 2, (det_face[1] + det_face[3]) / 2]
        )
        dist = np.linalg.norm(face_center - center)
        center_dist.append(dist)
    center_idx = center_dist.index(min(center_dist))
    return det_faces[center_idx], center_idx


class FaceRestoreHelper(object):
    """Helper for the face restoration pipeline (base class)."""

    def __init__(
        self,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        template_3points=False,
        device=None,
    ):
        self.template_3points = template_3points  #
        # the cropped face ratio based on the square face
        self.crop_ratio = crop_ratio  # (h, w)
        assert (
            self.crop_ratio[0] >= 1 and self.crop_ratio[1] >= 1
        ), "crop ration only supports >=1"
        self.face_size = (
            int(face_size * self.crop_ratio[1]),
            int(face_size * self.crop_ratio[0]),
        )

        if self.template_3points:
            self.face_template = np.array([[192, 240], [319, 240], [257, 371]])
        else:
            # standard 5 landmarks for FFHQ faces with 512 x 512
            self.face_template = np.array(
                [
                    [192.98138, 239.94708],
                    [318.90277, 240.1936],
                    [256.63416, 314.01935],
                    [201.26117, 371.41043],
                    [313.08905, 371.15118],
                ]
            )
        self.face_template = self.face_template * (face_size / 512.0)
        if self.crop_ratio[0] > 1:
            self.face_template[:, 1] += face_size * (self.crop_ratio[0] - 1) / 2
        if self.crop_ratio[1] > 1:
            self.face_template[:, 0] += face_size * (self.crop_ratio[1] - 1) / 2
        self.save_ext = save_ext

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # init face detection model
        self.face_det: RetinaFace = init_detection_model(
            det_model, half=False, device=self.device
        )

        # init face parsing model
        self.face_parse: ParseNet = init_parsing_model(
            model_name="parsenet", device=self.device
        )

    def get_crop_face(
        self,
        bathed_imgs: torch.Tensor,  # (B, H, W, C)
        only_keep_largest=False,
        only_center_face=False,
        resize=None,
        eye_dist_threshold=None,
        face_template_resize=None,
        face_template_x_offset=None,
        face_template_y_offset=None,
    ):
        if face_template_resize is None:
            face_template_resize = 1.0
        if face_template_x_offset is None:
            face_template_x_offset = 0.0
        if face_template_y_offset is None:
            face_template_y_offset = 0.0
        if resize is None:
            scale = 1
        else:
            h, w = bathed_imgs.shape[-2:]
            scale = min(h, w) / resize
            h, w = int(h / scale), int(w / scale)
            bathed_imgs = VF.resize(
                bathed_imgs, (w, h), interpolation=VF.InterpolationMode.BICUBIC
            )
        bathed_imgs = VF.normalize(bathed_imgs, [-1] * 3, [2] * 3).clamp(0, 1) * 255
        batched_bboxes = self.face_det.batched_detect_faces(bathed_imgs, 0.5) * scale
        affine_matrices = []
        cropped_faces = []
        find_face_idx = []
        bathed_imgs = (
            rearrange(bathed_imgs, "b c h w -> b h w c").detach().cpu().numpy()
        )
        for idx, (bboxes, img) in enumerate(zip(batched_bboxes, bathed_imgs)):
            landmarks = []
            det_faces = []
            for bbox in bboxes:
                # remove faces with too small eye distance: side faces or too small faces
                eye_dist = np.linalg.norm([bbox[5] - bbox[7], bbox[6] - bbox[8]])
                if eye_dist_threshold is not None and (eye_dist < eye_dist_threshold):
                    continue

                if self.template_3points:
                    landmark = np.array(
                        [[bbox[i], bbox[i + 1]] for i in range(5, 11, 2)]
                    )
                else:
                    landmark = np.array(
                        [[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)]
                    )
                landmarks.append(landmark)
                det_faces.append(bbox[0:5])
            if len(det_faces) == 0:
                continue
            if only_keep_largest:
                h, w, _ = img.shape
                det_faces, largest_idx = get_largest_face(det_faces, h, w)
                landmark = landmarks[largest_idx]
            elif only_center_face:
                h, w, _ = img.shape
                det_faces, center_idx = get_center_face(det_faces, h, w)
                landmark = landmarks[center_idx]
            else:
                landmark = landmarks[0]
            template = (
                np.stack(
                    [
                        self.face_template[:, 0] + face_template_x_offset,
                        self.face_template[:, 1] + face_template_y_offset,
                    ],
                    axis=1,
                )
                * face_template_resize
            )

            affine_matrix = cv2.estimateAffinePartial2D(
                landmark, template, method=cv2.LMEDS
            )[0]
            affine_matrices.append(affine_matrix)
            # warp and crop faces
            cropped_face = cv2.warpAffine(
                img,
                affine_matrix,
                self.face_size,
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(135, 133, 132),
            )  # gray
            cropped_faces.append(cropped_face.astype(np.float32))
            find_face_idx.append(idx)
        if len(cropped_faces) == 0:
            return None, None, None
        cropped_faces = VF.normalize(
            torch.from_numpy(np.stack(cropped_faces, axis=0))
            .permute(0, 3, 1, 2)
            .to(self.device)
            / 255.0,
            [0.5] * 3,
            [0.5] * 3,
        ).clamp(-1, 1)
        return cropped_faces, affine_matrices, find_face_idx

    def get_crop_face_from_affine_matrices(
        self, bathed_imgs: torch.Tensor, affine_matrices  # (B, H, W, C)
    ):
        if len(affine_matrices) == 0:
            return None
        bathed_imgs = VF.normalize(bathed_imgs, [-1] * 3, [2] * 3).clamp(0, 1) * 255
        cropped_faces = []
        bathed_imgs = (
            rearrange(bathed_imgs, "b c h w -> b h w c").detach().cpu().numpy()
        )
        for img, affine_matrix in zip(bathed_imgs, affine_matrices):
            # warp and crop faces
            cropped_face = cv2.warpAffine(
                img,
                affine_matrix,
                self.face_size,
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(135, 133, 132),
            )
            cropped_faces.append(cropped_face.astype(np.float32))
        cropped_faces = VF.normalize(
            torch.from_numpy(np.stack(cropped_faces, axis=0))
            .permute(0, 3, 1, 2)
            .to(self.device)
            / 255.0,
            [0.5] * 3,
            [0.5] * 3,
        ).clamp(-1, 1)
        return cropped_faces

    def get_inverse_affine(self, affine_matrices):
        """Get inverse affine matrix."""
        inverse_affine_matrices = []
        for affine_matrix in affine_matrices:
            inverse_affine = cv2.invertAffineTransform(affine_matrix)
            inverse_affine_matrices.append(inverse_affine)
        return inverse_affine_matrices

    def inverse_faces(self, restored_face_imgs, affine_matrices):
        face_parse = self.face_parse(restored_face_imgs)[0]
        face_parse = face_parse.argmax(dim=1).cpu().numpy()
        restored_face_imgs = (
            rearrange(
                VF.normalize(restored_face_imgs, [-1] * 3, [2] * 3).clamp(0, 1) * 255,
                "b c h w -> b h w c",
            )
            .cpu()
            .numpy()
        )
        inverse_affine_matrices = self.get_inverse_affine(affine_matrices)
        inv_restored_face_imgs = []
        inv_masks = []
        for restored_face, inverse_affine, parse in zip(
            restored_face_imgs, inverse_affine_matrices, face_parse
        ):
            mask = np.zeros(parse.shape)
            MASK_COLORMAP = [
                0,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
                0,
                0,
                0,
                0,
                0,
            ]
            for idx, color in enumerate(MASK_COLORMAP):
                mask[parse == idx] = color
            #  blur the mask
            mask = cv2.GaussianBlur(mask, (101, 101), 26)
            mask = cv2.GaussianBlur(mask, (101, 101), 26)
            # remove the black borders
            thres = 10
            mask[:thres, :] = 0
            mask[-thres:, :] = 0
            mask[:, :thres] = 0
            mask[:, -thres:] = 0
            mask = mask / 255.0
            h, w, _ = restored_face.shape
            inv_restored_face = cv2.warpAffine(
                restored_face, inverse_affine, (w, h), flags=cv2.INTER_CUBIC
            ).astype(np.float32)
            inv_mask = cv2.warpAffine(
                mask, inverse_affine, (w, h), flags=cv2.INTER_CUBIC
            ).astype(np.float32)
            inv_restored_face_imgs.append(inv_restored_face)
            inv_masks.append(inv_mask)
        inv_restored_face_imgs = VF.normalize(
            torch.from_numpy(np.stack(inv_restored_face_imgs, axis=0))
            .permute(0, 3, 1, 2)
            .to(self.device)
            / 255.0,
            [0.5] * 3,
            [0.5] * 3,
        ).clamp(-1, 1)
        inv_masks = (
            torch.from_numpy(np.stack(inv_masks, axis=0)).to(self.device).unsqueeze(1)
        )
        return inv_restored_face_imgs, inv_masks
