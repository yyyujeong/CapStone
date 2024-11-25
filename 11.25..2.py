import os
import cv2
import torch
import numpy as np
import albumentations
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from main import instantiate_from_config

# 기본 설정
STEPS = 100
IMG_PATH = ''  # 이미지 파일 경로
OUTPUT_PATH = ''  # 출력 경로
MARKER_SIZE = 20

# 디렉토리 생성
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 모델 로드 함수
def load_model():
    try:
        config = OmegaConf.load('models/ldm/inpainting_big/config.yaml')
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load('models/ldm/inpainting_big/last.ckpt')['state_dict'], strict=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Model loaded successfully. Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        return model, device
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")

# 전처리 함수
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    img = cv2.imread(image_path)
    if img.shape[0] < img.shape[1]:
        img = cv2.resize(img, (int(512 / img.shape[0] * img.shape[1]), 512))
    else:
        img = cv2.resize(img, (512, int(512 / img.shape[1] * img.shape[0])))
    img = albumentations.CenterCrop(height=512, width=512)(image=img)['image']
    return img

# 마스크 생성 함수
def create_mask(img):
    return np.ones((img.shape[0], img.shape[1]), dtype=np.float32)

# 원형 마커 드로잉 함수
def draw_circle(event, x, y, flags, param):
    global drawing, img, mask, ix, iy, MARKER_SIZE
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(img, (x, y), MARKER_SIZE, (0, 0, 0), -1)
        cv2.circle(mask, (x, y), MARKER_SIZE + 10, (0, 0, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# inpainting 함수
def inpaint(model, device, img, mask):
    masked_img = preprocess_for_inpainting(img, device)
    mask_input = preprocess_mask(mask, device)
    with torch.no_grad():
        c = model.cond_stage_model.encode(masked_img)
        cc = torch.nn.functional.interpolate(mask_input, size=c.shape[-2:])
        c = torch.cat((c, cc), dim=1)
        samples_ddim, _ = sampler.sample(S=STEPS, conditioning=c, batch_size=c.shape[0],
                                         shape=(c.shape[1] - 1,) + c.shape[-2:], verbose=False)
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        predicted_img = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        predicted_img = predicted_img.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
        predicted_img = cv2.cvtColor(predicted_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

        blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)
        inpainted_img = blurred_mask * img + (1 - blurred_mask) * predicted_img
        return inpainted_img.astype(np.uint8)

# 마스크된 이미지 전처리 함수
def preprocess_for_inpainting(img, device):
    masked_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    masked_img = np.expand_dims(masked_img, 0).transpose(0, 3, 1, 2)
    return torch.from_numpy(masked_img).to(device) * 2 - 1

# 마스크 전처리 함수
def preprocess_mask(mask, device):
    mask_input = np.expand_dims(mask, axis=(0, 1))
    return torch.from_numpy(mask_input).to(device) * -2 + 1

# 메인 실행 코드
if __name__ == '__main__':
    try:
        model, device = load_model()
        sampler = DDIMSampler(model)
        img = preprocess_image(IMG_PATH)
        img_ori = img.copy()
        mask = create_mask(img)

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_circle)

        while True:
            cv2.imshow('image', img)
            key = cv2.waitKey(1)

            if key == ord('q'):
                break
            elif key == ord('r'):
                img = img_ori.copy()
                mask = create_mask(img)
            elif key == ord('w'):
                inpainted_img = inpaint(model, device, img, mask)
                cv2.imshow('output', inpainted_img)
                filename = input("Enter filename to save: ")
                cv2.imwrite(os.path.join(OUTPUT_PATH, filename), inpainted_img)
            elif key == ord('+'):
                MARKER_SIZE += 5
                print(f"Marker size increased to {MARKER_SIZE}")
            elif key == ord('-') and MARKER_SIZE > 5:
                MARKER_SIZE -= 5
                print(f"Marker size decreased to {MARKER_SIZE}")

        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")