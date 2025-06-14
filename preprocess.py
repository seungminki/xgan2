import os
from PIL import Image
from rembg import remove
from facenet_pytorch import MTCNN
import torch

target_size = (224, 224) # 원하는 해상도

def preprocess(image_path: str, filename: str, output_folder: str):


    img = Image.open(image_path).convert("RGB")

    # 얼굴 감지기 초기화
    mtcnn = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')

    # 얼굴 bounding box 얻기
    boxes, probs = mtcnn.detect(img)

    if boxes is not None:
        for box, prob in zip(boxes, probs):
            if prob is None or prob < 0.90:  # 신뢰도 기준 (0.90 이상만 사용)
                print(f"{filename}: 얼굴 신뢰도 낮음 ({prob:.2f}), 건너뜀")
                continue
            
            x1, y1, x2, y2 = map(int, box)
            
            face_width = x2 - x1
            face_height = y2 - y1

            # 여유 padding
            padding_top = int(face_height * 0.4)
            padding_bottom = int(face_height * 0.4)
            padding_side = int(face_width * 0.4)

            x1_pad = x1 - padding_side
            y1_pad = y1 - padding_top
            x2_pad = x2 + padding_side
            y2_pad = y2 + padding_bottom

            # 정사각형 크기 결정
            center_x = (x1_pad + x2_pad) // 2
            center_y = (y1_pad + y2_pad) // 2
            half_side = max(x2_pad - x1_pad, y2_pad - y1_pad) // 2

            # 정사각형 범위 계산
            square_x1 = max(0, center_x - half_side)
            square_y1 = max(0, center_y - half_side)
            square_x2 = min(img.width, center_x + half_side)
            square_y2 = min(img.height, center_y + half_side)

            # 자르기
            cropped = img.crop((square_x1, square_y1, square_x2, square_y2))
            # cropped.save("face_square_crop.jpg")
            # save_path = os.path.join(output_folder, f"debug_crop_{filename}")
            # cropped.save(save_path)

            output_image = remove(cropped)

            # 2. 배경 흰색 합성
            white_bg = Image.new("RGB", output_image.size, (255, 255, 255))
            white_bg.paste(output_image, mask=output_image.split()[3])

            resized_img = white_bg.resize(target_size, Image.LANCZOS)

            # 저장 또는 바로 사용
            save_path = os.path.join(output_folder, f"result_{filename}")
            print(f"{filename} → 처리 완료 → {save_path}")
            resized_img.save(save_path)

    else:
        print(f"{filename} 얼굴을 찾을 수 없습니다.")

def make_and_process_files(folder_path, indices=[6, 7, 8], output_folder="korean_face/white_bg"):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        for i in indices:
            if filename.endswith(f"-{i}.jpg"):
                image_path = os.path.join(folder_path, filename)
                preprocess(image_path, filename, output_folder)
                break  # 일치하는 인덱스 하나만 처리하면 다음 파일로 넘어감

if __name__ == "__main__":
    input_folder = "/workspace/koreanface(201~250)"
    output_folder = "/workspace/korean_face_total_raw/white_bg"

    make_and_process_files(input_folder, output_folder=output_folder)