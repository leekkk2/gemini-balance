from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from app.core.constants import DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P


class SafetySetting(BaseModel):
    category: Optional[
        Literal[
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_CIVIC_INTEGRITY",
        ]
    ] = None
    threshold: Optional[
        Literal[
            "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
            "BLOCK_LOW_AND_ABOVE",
            "BLOCK_MEDIUM_AND_ABOVE",
            "BLOCK_ONLY_HIGH",
            "BLOCK_NONE",
            "OFF",
        ]
    ] = None


class GenerationConfig(BaseModel):
    stopSequences: Optional[List[str]] = None
    responseMimeType: Optional[str] = None
    responseSchema: Optional[Dict[str, Any]] = None
    candidateCount: Optional[int] = 1
    maxOutputTokens: Optional[int] = None
    temperature: Optional[float] = DEFAULT_TEMPERATURE
    topP: Optional[float] = DEFAULT_TOP_P
    topK: Optional[int] = DEFAULT_TOP_K
    presencePenalty: Optional[float] = None
    frequencyPenalty: Optional[float] = None
    responseLogprobs: Optional[bool] = None
    logprobs: Optional[int] = None
    thinkingConfig: Optional[Dict[str, Any]] = None
    # TTS相关字段
    responseModalities: Optional[List[str]] = None
    speechConfig: Optional[Dict[str, Any]] = None


class SystemInstruction(BaseModel):
    role: Optional[str] = "system"
    parts: Union[List[Dict[str, Any]], Dict[str, Any]]


class GeminiContent(BaseModel):
    role: Optional[str] = None
    parts: List[Dict[str, Any]]


class GeminiRequest(BaseModel):
    contents: List[GeminiContent] = []
    tools: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = []
    safetySettings: Optional[List[SafetySetting]] = Field(
        default=None, alias="safety_settings"
    )
    generationConfig: Optional[GenerationConfig] = Field(
        default=None, alias="generation_config"
    )
    systemInstruction: Optional[SystemInstruction] = Field(
        default=None, alias="system_instruction"
    )

    class Config:
        populate_by_name = True


class ResetSelectedKeysRequest(BaseModel):
    keys: List[str]
    key_type: str


class VerifySelectedKeysRequest(BaseModel):
    keys: List[str]


class GeminiEmbedContent(BaseModel):
    """嵌入内容模型

    parts 的值类型为 Any 而非 str：gemini-embedding-2 等原生多模态 embedding
    模型的 part 可以是 {"text": ...} 或 {"inline_data": {"mime_type": ...,
    "data": ...}} 这类嵌套对象，限定为 str 会让多模态请求 Pydantic 校验失败。
    """

    parts: List[Dict[str, Any]]


class GeminiEmbedRequest(BaseModel):
    """单一嵌入请求模型"""

    # 同时接受 camelCase 字段名与 snake_case 别名，避免不同调用方
    # （taskType/task_type、outputDimensionality/output_dimensionality）丢字段
    model_config = ConfigDict(populate_by_name=True)

    content: GeminiEmbedContent
    taskType: Optional[
        Literal[
            "TASK_TYPE_UNSPECIFIED",
            "RETRIEVAL_QUERY",
            "RETRIEVAL_DOCUMENT",
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
            "QUESTION_ANSWERING",
            "FACT_VERIFICATION",
            "CODE_RETRIEVAL_QUERY",
        ]
    ] = Field(default=None, alias="task_type")
    title: Optional[str] = None
    outputDimensionality: Optional[int] = Field(
        default=None, alias="output_dimensionality"
    )


class GeminiBatchEmbedRequest(BaseModel):
    """批量嵌入请求模型"""

    requests: List[GeminiEmbedRequest]
