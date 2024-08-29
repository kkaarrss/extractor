from pdf2image import convert_from_path
import os
import cv2
import numpy as np
import pytesseract
import base64
import httpx

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    # base_url="...",
    # organization="...",
    # other params...
)


def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text


def create(file_path):
    extracted_text = []

    # remove any folder and extension from the file name
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # retrieve the pages after page 26
    pages = convert_from_path(file_path)
    for i, page in enumerate(pages):
        success = False
        max_retries = 3
        retry_count = 0

        while not success and retry_count < max_retries:
            try:
                # Step 2: Preprocess the image (deskew)
                # preprocessed_image = deskew(np.array(page))
                page.save(f'image_converted_file_name_{i + 1}.png', 'PNG')

                image_converted = open(f'image_converted_file_name_{i + 1}.png', 'rb')

                # Step 3: Convert the image to base64
                image_data = base64.b64encode(image_converted.read()).decode("utf-8")
                # system_message = SystemMessage()
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": f"""Convert this pdf into markdown following these rules:
- IGNORE HEADERS AND FOOTERS.
- Convert any table to JSON format.
"""},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        }
                    ]
                )
                ai_msg = llm.invoke([(
                    "system",
                    "You are a powerful AI system that can convert PDFs to markdown.",
                ), message])

                text = ai_msg.content
                print(text, flush=True)

                with open(f"{file_name}.txt", "a") as file:
                    file.write(f"{text}\n")

                success = True
            except Exception as e:
                print(f"Error: {e}")
                retry_count += 1


if __name__ == "__main__":
    # for each file in the folder pdf call create(filename)
    for file in os.listdir("pdf"):
        if file.endswith(".pdf"):
            print(f"Processing file: {file}")
            create(f"pdf/{file}")
