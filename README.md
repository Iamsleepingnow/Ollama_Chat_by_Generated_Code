# Ollama_Chat_by_Generated_Code
Experimental project. Deepseek v3 generated code. You can running ollama LLM models already in your computer with a "nice" gradio GUI.

# How to Run
1. Keep python version >= 3.10.0.
2. Clone this repo into your disk.
3. Open CMD(Command Prompt) from repo path. `python -m venv .\venv\` to create virtual enviornment. If there r more than one python version in your computer, you need to replace `python` to selected one. For example: `D:\Softwares\python3\python.exe -m venv .\venv\`.
4. `.\venv\Scripts\activate` to activate virtual env, `python -m pip install -r .\requirements.txt` to install required modules. Then `deactivate` to exit env.
5. Open "Configs.json", replace the value of "model_name" key to your ollama model name. (How to check your ollama model list: `ollama list`) Then save and close the json file.
6. Double click "Launch.bat" to run.


# 简易Ollama对话机
实验性工程，使用Deepseek v3生成的源代码，能运行。可以使用本地已有的Ollama语言模型作为对话机，以及一个凑合的gradio界面。

# 如何运行
1. Python版本至少3.10.0。
2. 把仓库拉到本地。
3. 在刚拉的路径下打开CMD命令提示符，`python -m venv .\venv\`创建虚拟环境。如果安装有多个python版本，则需将`python`替换成指定版本。例如：`D:\Softwares\python3\python.exe -m venv .\venv\`。
4. 使用`.\venv\Scripts\activate`激活虚拟环境，`python -m pip install -r .\requirements.txt`安装依赖，最后`deactivate`退出环境。
5. 打开“Configs.json”，修改“model_name”键的值为你的Ollama模型代号。（如何查看已有的模型列表：`ollama list`）最后保存关闭。
6. 双击“Launch.bat”来运行。
