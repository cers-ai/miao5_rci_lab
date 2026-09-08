from typing import Literal
from pydantic import BaseModel, Field, model_validator


class Login(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class Password(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8, max_length=256)


class UserCreate(Login):
    role: Literal["管理员", "实验员"] = "实验员"
    enabled: bool = True


class UserUpdate(BaseModel):
    role: Literal["管理员", "实验员"]
    enabled: bool


class LocalModel(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    capability: Literal["流式语音识别", "说话人发现", "目标说话人识别", "语音活动检测"]
    load_mode: Literal["Python 模块", "命令行进程"] = "Python 模块"
    adapter: Literal["campp", "wespeaker", "pyannote", "vad", "paraformer", "command"]
    path: str = Field(min_length=1)
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    enabled: bool = True
    note: str = ""


class Service(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    provider: str = "OpenAI Compatible"
    base_url: str
    api_key: str | None = None
    timeout: int = Field(default=60, ge=1, le=600)
    models: list[str] = Field(min_length=1)
    default_model: str
    note: str = ""
    enabled: bool = True

    @model_validator(mode="after")
    def valid(self):
        from urllib.parse import urlparse
        u = urlparse(self.base_url)
        if u.scheme not in ("http", "https") or not u.hostname or u.username or u.password or u.query or u.fragment:
            raise ValueError("请填写不含密钥或查询参数的 HTTP(S) Base URL")
        if self.default_model not in self.models or any(not s.strip() for s in self.models):
            raise ValueError("默认模型必须属于可用模型列表")
        return self


class Agent(BaseModel):
    name: str = Field(min_length=1)
    service_id: str
    model: str
    system_prompt: str = Field(min_length=1, max_length=20000)
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=3000, ge=100, le=32000)


class E1(BaseModel):
    file_id: str
    model_id: str = "campp"
    speaker_count: int | None = Field(default=None, ge=1, le=30)
    clustering_threshold: float = Field(default=0.65, ge=0, le=2)
    min_speech_duration: float = Field(default=0.5, ge=0.2, le=5)


class Profile(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    run_id: str
    speaker_ids: list[str] = Field(min_length=1)
    excluded_segments: list[str] = []
    model_id: str = "wespeaker"


class E2(BaseModel):
    profile_id: str
    file_ids: list[str] = Field(min_length=1, max_length=30)
    model_id: str = "wespeaker"
    similarity_threshold: float = Field(default=0.65, ge=-1, le=1)


class Interval(BaseModel):
    start_ms: float = Field(ge=0)
    end_ms: float = Field(gt=0)

    @model_validator(mode="after")
    def valid(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("结束时间必须晚于开始时间")
        return self


class E3(BaseModel):
    profile_id: str
    file_id: str | None = None
    lock_threshold: float = Field(default=0.65, ge=-1, le=1)
    unlock_threshold: float = Field(default=0.5, ge=-1, le=1)
    confirm_windows: int = Field(default=2, ge=1, le=10)
    ground_truth: list[Interval] | None = None

    @model_validator(mode="after")
    def valid(self):
        if self.unlock_threshold >= self.lock_threshold:
            raise ValueError("解锁阈值必须低于锁定阈值")
        return self


class Review(BaseModel):
    item_id: str
    verdict: Literal["待核验", "正确", "同一人被拆成多个Speaker", "不同人被错误合并", "其它问题", "误识别", "漏识别"]
    note: str = Field(default="", max_length=5000)


class GroundTruth(BaseModel):
    intervals: list[Interval]


class Selection(BaseModel):
    run_ids: list[str] = Field(min_length=1, max_length=100)
    analysis_id: str | None = None


class Analysis(BaseModel):
    summary: str
    findings: list[str]
    recommended_configuration: dict
    parameter_observations: list[str]
    risks: list[str]
    next_experiments: list[str]
