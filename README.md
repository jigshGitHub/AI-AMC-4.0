# AI-AMC-4.0

Repository scaffold for the AI-AMC-4.0 project.

This repo's `.gitignore` was updated to include ignores for common OS/editor artifacts, Python/Node files, and .NET / Visual Studio files (bin/, obj/, .vs/, TestResults/, NuGet packages, local.settings.json, etc.).

Quick notes
- Use a virtual environment for Python (e.g. `python -m venv .venv`) and do not commit it — `.gitignore` already excludes common venv folders.
- For .NET development, build outputs (`bin/`, `obj/`) and IDE files (`.vs/`, user-specific `.user` files) are ignored.
- If you use Azure Functions locally, keep secrets out of source control — `local.settings.json` is ignored.

If you want the README to include project-specific setup instructions (build steps, tests, how to run locally), tell me what stack(s) you're using and I will add a short Getting Started section.
