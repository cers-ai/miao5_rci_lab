# 妙悟 · 实时认知智能实验室 V0.1

依据项目原型和开发方案实现的本地实验工作台。React + TypeScript + Vite 前端，FastAPI + SQLite 后端。完整范围为系统基础能力、模型设置和 A01 四步实验；其它能力保留设计占位。

## 启动

当前项目目录：`D:\miao5_rci_lab`。

```powershell
# 首次初始化（系统需安装 uv、Node.js 22+、ffmpeg/ffprobe）
python scripts/init.py
cd frontend
npm ci
cd ..

# 启动前后端隐藏进程，日志写入 workspace/logs
powershell -ExecutionPolicy Bypass -File scripts/start.ps1

# 停止本项目启动的进程
powershell -ExecutionPolicy Bypass -File scripts/stop.ps1
```

- 前端固定：[http://localhost:3000](http://localhost:3000)，端口占用会明确报错，不会自动换端口。
- 后端：[http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)。前端经同源代理访问 API 和 WebSocket。
- 默认账号：`admin`；初始密码：`abcd@1234`。系统设置 → 用户与账号可修改密码。
- API 文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)。业务接口需要登录 Cookie。

也可分别在两个终端启动：

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
# 第二个终端
cd frontend
npm run dev
```

Python 使用 3.11；`uv.lock` 和 `frontend/package-lock.json` 固定依赖。复现时执行 `uv sync --locked --python 3.11` 和 `npm ci`。生产构建检查：`cd frontend; npm run build`。

## 模型

模型统一存放在 `workspace/models/`，各目录包含 `manifest.json`（来源、固定版本、大小、SHA256）。可先执行不依赖ML运行环境的下载器：

```powershell
python scripts/model_fetch.py
# 注册、校验并真实推理测试；pyannote 未授权将显示未安装
.venv\Scripts\python.exe scripts/download_models.py
# 单独重新下载/测试
.venv\Scripts\python.exe scripts/download_models.py --only campp
```

| 注册模型 | 固定来源/版本 | 实际用途 |
|---|---|---|
| CAM++ 中文 | iic/speech_campplus_sv_zh-cn_16k-common / v2.0.2 | E1声纹与聚类，也可E2建档 |
| FSMN VAD | iic/speech_fsmn_vad_zh-cn-16k-common-pytorch / v2.0.4 | E1/E2语音切分和E3活动检测 |
| WeSpeaker 中文 ResNet34-LM | wenet官方预训练模型 / cnceleb_resnet34_LM.onnx | E1聚类、E2识别、E3滑窗声纹 |
| Paraformer Streaming | iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online / v2.0.4 | 原型预设流式模型，支持真实流式加载测试；A02不在V0.1范围 |
| pyannote Community-1 | pyannote/speaker-diarization-community-1 / 授权后固定commit | E1完整说话人分离 |

WeSpeaker 使用官方 ONNX checkpoint 和官方80维 hamming fbank + CMN前端，由 Python ONNX Runtime 本地执行。原型的 `wespeaker-ecapa` 是占位标识，正式界面显示实际下载的中文 ResNet34-LM。

pyannote 按用户确认保留待授权：前往[官方模型页面](https://huggingface.co/pyannote/speaker-diarization-community-1)接受模型条件，然后在本机设置 `HF_TOKEN`，执行 `--only pyannote`。Token不写入数据库、不写入模型清单。授权后重新启动后端。不能访问受限权重时不会生成模拟分离结果。

模型在初始化及后端启动时运行真实中文语音样本测试；测试通过后才标记“可用”。可以在系统设置编辑设备、路径、加载方式并启用/停用。当前默认锁定 CPU 环境；选择 CUDA 而没有 CUDA 时会明确失败。

## 完成一次真实实验

1. 登录，确认 CAM++、VAD、WeSpeaker 状态为“可用”。
2. A01 → 01：上传会议录音或选择历史文件，配置模型、人数或距离阈值、最短语音，开始实验。后台排队执行，状态实时刷新。每轮独立保存，支持取消。
3. 试听每个 Speaker 的真实片段，保存拆分/合并/正确等人工核验；调整配置再次运行。
4. A01 → 02：选择已完成的 E1，勾选目标 Speaker（支持合并），排除错误片段，填写人物名称、建档模型。重叠片段自动排除。最多采集60秒，少于3秒拒绝，少于30秒记录警告。
5. 选择目标声纹及多个测试录音，设置相似度阈值运行 E2。可查看并试听全部目标及非目标片段，核验误识别与漏识别。相似度是余弦值[-1,1]，不是概率；更换模型需重新建档。
6. A01 → 03：选目标、锁定和解锁阈值、连续确认窗口数，选择真实时间回放或麦克风。麦克风使用 AudioWorklet 采集、16kHz PCM16 WebSocket传输，停止后保存原始声音和事件。现场可标记目标开始/结束和误锁。
7. 回放完成后以秒填写 Ground Truth，每行 `开始,结束`，保存以计算正式指标。留空保存表示“整段没有目标”，不同于“没有人工标注”。
8. A01 → 04：选择代表性 Run，调用配置好的模型服务进行综合分析，生成并下载 MD、CSV、JSON 报告。

实验数据仓可重新试听录音、查看每轮详情、重新下载报告。已被实验/声纹引用的文件禁止删除，防止证据丢失。未引用文件可在确认后删除。

## 辅助分析服务

系统设置 → 模型注册 → 注册模型服务，填写 OpenAI Compatible Base URL、API Key、可用模型及默认模型。测试会真实调用 `GET /models` 和 `POST /chat/completions`。随后在“辅助分析智能助手”中选择服务、模型、System Prompt、Temperature、Max Tokens。

API Key在本地数据库中加密保存，密钥文件为 `workspace/.service-key`；列表只返回是否配置。编辑时留空保留，勾选清除才删除已有密钥。备份数据库时需要一起备份密钥文件。

综合分析发送所选记录的文件名称、参数、模型版本、推理输出和人工核验；不会发送音频或声纹向量。调用失败、超时或非结构化返回会明确报错。无模型服务时可生成“未执行智能分析”的实验证据汇总。隔离HTTP测试服务只用于测试协议，不替代真实LLM分析。

## 实时指标口径

- 所有输入使用1.5秒滑窗、0.5秒步长，锁定须达到连续确认窗口数，已锁定时低于解锁阈值或没有近期语音则解锁。
- 时间线保存音频位置、实际决策时间、推理耗时、相似度、状态和Lock/Unlock事件。
- 首次锁定时延：首个目标标注区间开始至实际锁定；未锁定为 `null`。
- 误锁：发生在所有目标标注区间之外的Lock事件次数。
- 解锁时延：目标区间结束至解锁（多个有效值取平均）；未解锁或目标持续到录音结束时不伪造测量。
- 重新锁定时延：第二个及之后目标区间开始到重新锁定，保留每段明细与漏锁数。
- 覆盖率：实际LOCKED区间与目标标注区间交集时长 / 目标总时长，不把锁定决策倒填至声纹滑窗起点。
- 无Ground Truth只显示运行观测值，正式准确率、误锁和时延指标为空。

## 本地命令行模型协议

管理员可以注册“命令行进程”加载方式和 `command` 适配器，路径须为实际可执行文件。后台不经过 shell，将一行JSON传到stdin：

```json
{"operation":"embedding","audio_path":"绝对路径/input.wav","sample_rate":16000,"parameters":{}}
```

进程须在120秒内从stdout返回 `{"embedding":[有限浮点数...]}`，日志写stderr，错误使用非0退出码。Embedding维数至少8且范数非0。可用于E1 VAD+聚类、E2建档/识别和E3；API不会接受任意Python代码或命令字符串。Python模块加载通过明确适配器选择已有受支持模型结构。

## 数据与备份

`workspace/lab.sqlite3` 存储用户、会话、模型配置、实验记录、核验、分析和报告。`uploads/`保存原始与16kHz单声道版本；`target_profiles/`保存声纹；`experiment_outputs/`保存推理原始JSON；`reports/`保存不可变报告快照。备份整个workspace可保留模型、音频及证据。

服务默认只监听本机。修改端口/监听地址、部署到局域网或公网时应另行配置HTTPS及访问控制，麦克风需要安全浏览器上下文（localhost支持）。

## 验证

```powershell
.venv\Scripts\python.exe -m pytest -q --junitxml=workspace/logs/tests.xml
.venv\Scripts\python.exe scripts/health_check.py
cd frontend
npm run build
npx playwright install chromium
npm run test:e2e
```

测试使用临时数据库，避免将隔离协议测试数据写进正式工作区。浏览器完整流程会留下有明确测试名称的真人语音样本、实验和报告，便于验查。真人公开语音组合验证仅证明推理与保存链路，不代替实际多人会议、重叠讲话或跨设备的质量验收。

详细实施状态见 `开发计划.md`；实测结果与限制见 `测试报告.md`。

## 官方模型与技术来源

- [CAM++ 官方模型](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common)
- [FunASR 官方实现](https://github.com/modelscope/FunASR)
- [WeSpeaker 官方模型下载实现](https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/cli/hub.py)
- [WeSpeaker 官方特征前端](https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/cli/speaker.py)
- [pyannote Community-1 官方模型](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [TorchCodec 与 PyTorch 兼容表](https://github.com/pytorch/torchcodec#compatibility-with-torch-versions)
