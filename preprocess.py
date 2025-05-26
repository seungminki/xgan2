import os
from PIL import Image
from rembg import remove
from facenet_pytorch import MTCNN

target_size = (224, 224) # 원하는 해상도

input_folder = "/workspace/google_data/raw"
output_folder = "/workspace/google_data/white_bg"
os.makedirs(output_folder, exist_ok=True)

def main(image_path: str, filename: str):

    img = Image.open(image_path)

    # 얼굴 감지기 초기화
    mtcnn = MTCNN(keep_all=False)

    # 얼굴 bounding box 얻기
    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        for box in boxes:
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
        print("얼굴을 찾을 수 없습니다.")


for filename in os.listdir(input_folder):

    image_path = os.path.join(input_folder, filename)
    main(image_path, filename)