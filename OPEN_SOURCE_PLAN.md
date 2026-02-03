# paper-reviewer-v2 开源规划

本文档帮助你系统化地准备并发布 **EXPERMATCH** 代码仓库为开源项目。

---

## 一、当前状态概览

| 项目 | 状态 | 说明 |
|------|------|------|
| README | ✅ 已有 | 结构清晰，含任务、数据集、模型、环境、引用 |
| 根目录 LICENSE | ⚠️ 缺失 | README 声明 MIT，但仓库根目录无 LICENSE 文件 |
| .gitignore | ⚠️ 缺失 | 易误提交 `__pycache__`、虚拟环境、大模型文件等 |
| 第三方代码 (CoF-main) | ✅ 已标注 | README 中已说明来源与引用 |
| 引用占位符 | ⚠️ 待更新 | `your-org`、`Anonymous` 需在发表/开源时替换 |
| 预训练模型文件 | ⚠️ 需决策 | `models/` 下含大量 .bin/.safetensors，需决定是否进仓库 |

---

## 二、开源前必做清单

### 1. 法律与许可

- [ ] **添加根目录 LICENSE 文件**  
  - README 已声明 MIT，在仓库根目录添加 `LICENSE`（MIT 全文），并填上版权年份与版权方名称。
- [ ] **确认第三方许可兼容**  
  - `CoF-main/` 为 Apache 2.0，与 MIT 主仓库可并存；在 README 或 NOTICE 中保留对 CoF 的 attribution 即可。
- [ ] **数据集许可与引用**  
  - 在 README 或 `data/README.md` 中注明各数据集（SIGIR、KDD、NIPS、SciRepEval、Stelmakh 等）的来源、许可及引用方式，避免再分发违规。

### 2. 仓库内容与 .gitignore

- [ ] **添加 .gitignore**  
  - 至少包含：`__pycache__/`、`*.pyc`、`.env`、`venv/`、`*.egg-info/`、`.idea/`、`*.log`。  
  - 若模型不放入 Git：将 `models/` 下大文件或整个 `models/` 加入忽略，并改为通过脚本/文档下载。
- [ ] **大文件策略二选一**  
  - **方案 A**：不把预训练权重放进 Git，在 README 或 `scripts/download_models.sh` 中提供 Hugging Face / 网盘下载链接与步骤。  
  - **方案 B**：若必须用 Git 托管权重，使用 **Git LFS** 管理 `*.bin`、`*.safetensors`、`*.ckpt`、`*.pth` 等，并在 README 中说明需安装 Git LFS。

### 3. 文档与占位符

- [ ] **替换引用与链接**  
  - 将 README 中 `https://github.com/your-org/paper-reviewer-v2` 改为实际仓库 URL。  
  - 论文发表后，将 BibTeX 中 `Anonymous` 改为真实作者与会议信息。
- [ ] **（可选）数据说明**  
  - 在 `data/README.md` 中简述各目录结构、`*_meta.json` 含义、以及如何添加新数据集，便于社区复现与扩展。

### 4. 敏感与隐私

- [ ] **检查敏感信息**  
  - 搜索 API Key、密码、内网地址、真实姓名/邮箱（若不应公开）。  
  - 确保无公司内部路径、未脱敏日志等。
- [ ] **论文草稿**  
  - `ExpertiseMatch__A_Unified_Benchmark_...txt` 若为审稿中稿件，可考虑从公开仓库移除或放入私有分支，避免“匿名审稿”泄露。

### 5. 体验与复现

- [ ] **环境与依赖**  
  - 已提供 `requirements.txt`、`setup.sh`、多环境说明；开源前再跑一遍 `setup.sh` 与 `run_experiment.sh` 的示例，确保文档与脚本一致。
- [ ] **Issue/PR 模板（可选）**  
  - 在 GitHub 仓库的 `.github/ISSUE_TEMPLATE/` 和 `.github/PULL_REQUEST_TEMPLATE.md` 增加简单模板，便于用户反馈与贡献。

---

## 三、建议时间线

1. **第 1 步**：完成「开源前必做清单」中的法律、.gitignore、大文件策略和占位符替换。  
2. **第 2 步**：内部或小范围试用一次「克隆 → setup → 跑通 README 示例」，修掉文档/路径/依赖问题。  
3. **第 3 步**：创建 GitHub（或 GitLab）仓库，推送代码，将 README 中的链接改为新仓库地址。  
4. **第 4 步**：（可选）在论文/社交媒体注明「代码已开源」，并附仓库链接与引用。

---

## 四、本仓库已为你准备的内容

- **OPEN_SOURCE_PLAN.md**（本文档）：开源规划与检查项。  
- **LICENSE**：根目录 MIT 许可证（需你补全版权方名称与年份）。  
- **.gitignore**：Python + 常见 IDE/缓存/虚拟环境；未默认忽略 `models/`，你可根据「大文件策略」自行取消注释或增加规则。

完成上述清单后，即可按你的时间线对外公开仓库。若你希望，我也可以根据你选的「大文件策略」帮你写一版 `download_models.sh` 或 Git LFS 说明片段。
