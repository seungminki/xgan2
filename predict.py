from utils import configure_model
from models import Avatar_Generator_Model

def predict(model_config_path: str, input_image: str):
    # 1. 설정 파일 로드
    config = configure_model("config.json", use_wandb=False)

    config.model_path = model_config_path
    print(config.model_path)

    # 2. 모델 인스턴스 생성 + 가중치 로드
    model = Avatar_Generator_Model(config, use_wandb=False)
    model.load_weights(config.model_path)

    # 3. 입력 이미지 경로 (실제 사진)
    input_image_path = f"/workspace/image/input/{input_image}"

    # 4. 출력 이미지 경로
    output_image_path = f"/workspace/image/output/{input_image}"

    # 5. 추론 수행
    cartoon_pil_image, _ = model.generate(input_image_path, output_image_path)

    # 6. 결과 보기
    cartoon_pil_image.show()

if __name__ == "__main__":
    predict("weights/config17_epochs400", "1.jpg")