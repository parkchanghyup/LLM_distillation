import chainlit as cl
import httpx
import json
from typing import Dict

# API 설정
API_URL = "http://localhost:8000/summarize"

@cl.on_chat_start
async def start():
    """채팅 시작시 실행되는 함수"""
    await cl.Message(
        content="안녕하세요! 텍스트 요약 AI 어시스턴트입니다. 요약하고 싶은 텍스트를 입력해주세요."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """사용자 메시지 처리 함수"""
    # 로딩 메시지 표시
    msg = cl.Message(content="요약문을 생성하고 있습니다...")
    await msg.send()

    try:
        # API 요청
        async with httpx.AsyncClient() as client:
            response = await client.post(
                API_URL,
                json={"text": message.content},
                timeout=30.0
            )
            response.raise_for_status()
            result: Dict = response.json()

        # 요약 결과 표시
        await msg.update(content=f"요약문:\n\n{result['summary']}")

        # 원본 텍스트와 요약문 비교를 위한 요소 추가
        elements = [
            cl.Text(name="original", content=message.content, display="side", language="text"),
            cl.Text(name="summary", content=result['summary'], display="side", language="text")
        ]
        await msg.update(elements=elements)

    except httpx.HTTPError as e:
        error_message = f"API 요청 중 오류가 발생했습니다: {str(e)}"
        await msg.update(content=error_message)
    except Exception as e:
        error_message = f"오류가 발생했습니다: {str(e)}"
        await msg.update(content=error_message)
