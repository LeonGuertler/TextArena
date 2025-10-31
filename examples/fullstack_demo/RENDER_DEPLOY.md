# Render 部署指南

## 快速部署步骤

### 方式一：使用 render.yaml（推荐）

1. 确保代码已推送到 GitHub（branch: `feature/web-version`）
2. 登录 [Render Dashboard](https://dashboard.render.com/)
3. 点击 "New +" → "Blueprint"
4. 连接 GitHub 仓库，选择 `feature/web-version` branch
5. Render 会自动检测并读取 `render.yaml` 配置
6. 在 Environment 部分添加以下环境变量：
   - `SUPABASE_URL` - 你的 Supabase 项目 URL
   - `SUPABASE_SERVICE_ROLE_KEY` - Supabase service role key
   - `SUPABASE_ANON_KEY` - Supabase anon key
   - `SUPABASE_JWT_SECRET` - Supabase JWT secret（在 Supabase Dashboard → Settings → API → JWT Secret）
   - `OPENAI_API_KEY` - 你的 OpenAI API key

### 方式二：手动配置

1. 登录 Render Dashboard
2. 点击 "New +" → "Web Service"
3. 连接 GitHub 仓库，选择 `feature/web-version` branch
4. 配置如下：
   - **Name**: textarena-fullstack-demo
   - **Environment**: Python 3
   - **Build Command**: `pip install -e .`
   - **Start Command**: `python examples/fullstack_demo/main.py`
5. 添加环境变量（同上）

## 本地测试

本地仍使用原有方式启动，不受影响：

```bash
uv run examples/fullstack_demo/main.py
```

## 注意事项

- Render 会自动设置 `PORT` 环境变量
- 生产环境会自动禁用 reload（通过 `FULLSTACK_DEMO_RELOAD=0`）
- 确保 Supabase 表 `game_runs` 已创建（参考 README.md）

