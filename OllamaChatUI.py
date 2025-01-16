import json
import random
import time
import atexit
import os
import ollama
import gradio as gr
from datetime import datetime
import webbrowser
import shutil

# 默认模型配置信息（参考）一般情况下会被./Configs.json覆盖
configs = {
    # IP地址
    "host": "127.0.0.1",
    # 端口号
    "port": "11434",
    # UI端口
    "uiport": "7866",
    # 模型名称
    "model_name": "qwq-7b",
    # 模型参数选项
    "options": {
        "temperature": 1.0,  # temperature值越高，模型创造性越强；值越低，模型相干性越强。值1为默认值。
        "num_ctx": 4096,  # num_ctx是上下文最大token数量，默认为4096。
        "top_k": 50,  # top_k采样会影响回答的多样化程度，推荐范围10~100，默认40。
        "top_p": 0.9,  # top_P相当于top_K的阈值，也会影响多样化程度，推荐范围0.5~0.95，默认0.9。
        "repeat_penalty": 1.2,  # repeat_penalty是重复惩罚，能避免生成重复信息，推荐范围0.9~1.5，默认1.1。
        "seed": -1  # seed是使用固定种子，-1为使用随机种子。
    }
}

# 默认系统提示词和助手第一句话
DEFAULT_SYSTEM_PROMPT = "## Role:\n我是一个乐于助人的智能助手，我精通多国语言，对科学、文学、历史、数学、哲学、艺术等文化领域了如指掌。请告诉我你的需求，我都能尽我所能地执行并给出建议。"
DEFAULT_ASSISTANT_FIRST_PROMPT = "我是你的小助手，你可以叫我小助，有什么需要帮助的吗？😋"

# 加载配置
def load_config():
    config_file = './Configs.json'
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                conf = json.load(f)
                configs.update(conf)
        except Exception as e:
            print(f"加载配置文件失败: {e}，使用默认配置")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(configs, f, indent=4, ensure_ascii=False)
    else:
        print("配置文件不存在，创建默认配置")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(configs, f, indent=4, ensure_ascii=False)

# 加载系统提示词
def load_system_prompt():
    system_prompt_file = './SystemPrompt.md'
    if os.path.exists(system_prompt_file):
        try:
            with open(system_prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"加载系统提示词失败: {e}，使用默认提示词")
            with open(system_prompt_file, 'w', encoding='utf-8') as f:
                f.write(DEFAULT_SYSTEM_PROMPT)
            return DEFAULT_SYSTEM_PROMPT
    else:
        print("系统提示词文件不存在，创建默认提示词")
        with open(system_prompt_file, 'w', encoding='utf-8') as f:
            f.write(DEFAULT_SYSTEM_PROMPT)
        return DEFAULT_SYSTEM_PROMPT

# 加载助手第一句话
def load_assistant_first_prompt():
    assistant_first_prompt_file = './AssistantFirstPrompt.md'
    if os.path.exists(assistant_first_prompt_file):
        try:
            with open(assistant_first_prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"加载助手第一句话失败: {e}，使用默认提示词")
            with open(assistant_first_prompt_file, 'w', encoding='utf-8') as f:
                f.write(DEFAULT_ASSISTANT_FIRST_PROMPT)
            return DEFAULT_ASSISTANT_FIRST_PROMPT
    else:
        print("助手第一句话文件不存在，创建默认提示词")
        with open(assistant_first_prompt_file, 'w', encoding='utf-8') as f:
            f.write(DEFAULT_ASSISTANT_FIRST_PROMPT)
        return DEFAULT_ASSISTANT_FIRST_PROMPT

# 加载配置和提示词
load_config()
system_prompt = load_system_prompt()
assistant_first_prompt = load_assistant_first_prompt()

# 对话历史
history = []

# 全局标志，用于中断模型输出
stop_generation = False

# 模型历史重置
def model_history_restart():
    history.clear()
    # 添加系统提示词和助手的第一句话
    history.append({"role": "system", "content": system_prompt})
    history.append({"role": "assistant", "content": assistant_first_prompt})
    return history

def chat_to_ollama(user_input):  # 与ollama聊天，返回聊天迭代器
    global stop_generation
    stop_generation = False
    history.append({"role": "user", "content": user_input})
    response = ollama.Client(host=f"http://{configs['host']}:{configs['port']}").chat(
        model=configs['model_name'], stream=True, messages=history, options={
            "temperature": configs['options']['temperature'],
            "num_ctx": configs['options']['num_ctx'],
            "top_k": configs['options']['top_k'],
            "top_p": configs['options']['top_p'],
            "repeat_penalty": configs['options']['repeat_penalty'],
            "seed": configs['options']["seed"] if configs['options']["seed"] >= 0 else random.randint(0, 1000000000)})

    assistant_response = ""
    for chunk in response:
        if stop_generation:
            break
        assistant_response += chunk['message']['content']
        yield assistant_response
    if not stop_generation:
        history.append({"role": "assistant", "content": assistant_response})

def stop_at_exit():  # 退出
    # 当用户退出应用时，执行控制台命令：ollama stop <模型名称>
    os.system(f'ollama stop {configs["model_name"]}')
    print('\n<- 正在退出 ->\n')
    time.sleep(1.5)

def load_history(file):
    try:
        with open(file.name, 'r', encoding='utf-8') as f:
            loaded_history = json.load(f)
        history.clear()
        history.extend(loaded_history)
        return loaded_history
    except Exception as e:
        print(f"加载历史失败: {e}")
        return None

def update_config(temperature, top_k, top_p, repeat_penalty, seed):
    configs['options']['temperature'] = temperature
    configs['options']['top_k'] = top_k
    configs['options']['top_p'] = top_p
    configs['options']['repeat_penalty'] = repeat_penalty
    configs['options']['seed'] = seed

def random_seed():
    return random.randint(0, 1000000000)

def stop_generation_fn():
    global stop_generation
    stop_generation = True

# 获取历史文件列表
def get_history_files():
    history_dir = './history'
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    files = os.listdir(history_dir)
    return [f for f in files if f.endswith('.json')]

# 读取历史文件并更新聊天记录
def load_history_from_dropdown(selected_file):
    if selected_file == "无":
        # 如果选择“无”，则重置历史数组和 Chatbot 组件
        return model_history_restart()
    else:
        history_dir = './history'
        filepath = os.path.join(history_dir, f"{selected_file}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    loaded_history = json.load(f)
                history.clear()
                history.extend(loaded_history)
                return loaded_history
            except Exception as e:
                print(f"加载历史失败: {e}")
        return None

# 刷新历史文件列表
def refresh_history_files():
    model_history_restart()  # 重置历史数组
    files = get_history_files()
    # 添加默认选项“无”，并确保它是第一个选项
    return gr.Dropdown(choices=["无"] + [os.path.splitext(f)[0] for f in files], value="无")

# 保存历史时使用前缀
def save_history_with_prefix(prefix):
    if not prefix.strip():  # 如果前缀为空，则使用默认值
        prefix = "History"
    # 确保 history 目录存在
    history_dir = '.\\history\\'
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)

    # 生成随机文件名
    random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{prefix}_{current_time}_{random_str}.json"
    filepath = os.path.join(history_dir, filename)

    # 保存历史到文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    return filepath

# 打开历史文件夹
def open_history_folder():
    history_dir = '.\\history\\'
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    # 打开文件夹
    if os.name == 'nt':  # Windows
        os.startfile(history_dir)
    elif os.name == 'posix':  # macOS 或 Linux
        os.system(f'open "{history_dir}"' if os.uname().sysname == 'Darwin' else f'xdg-open "{history_dir}"')

# Gradio界面
with gr.Blocks(
        title=f"🔥 对话机：{configs['model_name']} 🔥",
        css=".gradio-container {background-color: #252A34;}"
            ".chatbot {background-color: #08D9D6; overflow: auto;}"
            ".top-panel {display: none;}"
            ".btn-submit {background-color: #FF2E63; height: 95px; max_height: 95px; font-size: 30px;}"
            ".btn-normal {background-color: #08D9D6;}"
            ".btn-refresh {height: 100px;}"
            ".txt-chat-input {height: 150px; max_height: 150px;}"
) as demo:
    # 标题
    gr.HTML("<h1 align='center' class='normal-text'>🔥 Ollama 对话机器人 🔥</h1>")
    # 聊天界面
    chatbot = gr.Chatbot(type="messages", label=f"{configs['model_name']}：", value=model_history_restart(),
                         editable="all", height=500,
                         show_copy_button=True, avatar_images=(None, None), resizeable=True,
                         min_height=200)  # 初始化时加载助手的第一句话
    with gr.Row():
        msg = gr.Textbox(label="输入消息", container=False, lines=7, max_lines=20,
                         placeholder=">>> 说点什么吧", scale=6, elem_classes="txt-chat-input")
        # 功能按钮
        with gr.Column(scale=1):
            submit_btn = gr.Button("提交", variant="primary", scale=1, elem_classes="btn-submit")
            stop_btn = gr.Button("中止", scale=1, elem_classes="btn-normal")

    # 文件保存组件
    file_save = gr.File(visible=False)

    # 历史清单
    with gr.Accordion("历史清单", open=False):
        gr.HTML("<p align='left' style='color: #08D9D6;'>“历史清单”允许你查看和管理以前的对话历史。\n"
                "点击“刷新历史记录”可以重新加载历史清单，通过选择历史清单来加载历史。"
                "凡按下“保存历史”、“刷新历史记录”都会清除当前对话历史。</h1>")
        with gr.Row():
            prefix_input = gr.Textbox(label="历史文件前缀", value="HistoryFile", placeholder="请输入保存历史文件的前缀",
                                      lines=1, max_lines=1, container=False)
            save_btn = gr.Button("保存历史", elem_classes="btn-normal", min_width=15)
            load_btn = gr.UploadButton("读取历史", file_types=[".json"], elem_classes="btn-normal", min_width=15)
            open_folder_btn = gr.Button("打开历史文件夹", elem_classes="btn-normal", min_width=15)
        with gr.Row():
            history_dropdown = gr.Dropdown(choices=["无"], value="无", label="选择历史文件", scale=3)
            refresh_btn = gr.Button("刷新历史目录 & 清除历史", elem_classes="btn-refresh", scale=1, min_width=35)

    # 模型配置
    with gr.Accordion("模型配置", open=False):
        temperature = gr.Slider(0.0, 2.0, value=configs['options']['temperature'], label="Temperature",
                                info="temperature值越高，模型创造性越强；值越低，模型相干性越强。")
        top_k = gr.Slider(10, 100, value=configs['options']['top_k'], step=1, label="Top_k",
                          info="top_k采样会影响回答的多样化程度。")
        top_p = gr.Slider(0.5, 0.95, value=configs['options']['top_p'], label="Top_p",
                          info="top_P相当于top_K的阈值，也会影响多样化程度。")
        repeat_penalty = gr.Slider(0.8, 1.5, value=configs['options']['repeat_penalty'], label="Repeat penalty",
                                   info="repeat_penalty是重复惩罚，能避免生成重复信息。")
        with gr.Row():
            seed = gr.Number(value=configs['options']['seed'], label="Seed",
                             info="seed是使用固定种子，-1为使用随机种子。", scale=5)
            random_seed_btn = gr.Button("随机", scale=1, elem_classes="btn-refresh")


    # 交互逻辑
    def respond(user_input, chat_history):
        bot_response = chat_to_ollama(user_input)
        chat_history.append({"role": "user", "content": user_input})  # 添加用户输入到聊天记录
        for response in bot_response:
            if len(chat_history) > 0 and chat_history[-1]["role"] == "user":
                chat_history.append({"role": "assistant", "content": response})  # 添加助手响应
            else:
                chat_history[-1]["content"] = response  # 更新助手响应
            yield chat_history

    # 交互逻辑
    submit_btn.click(respond, [msg, chatbot], [chatbot])
    stop_btn.click(stop_generation_fn, None, None)
    save_btn.click(save_history_with_prefix, prefix_input, [file_save]).then(refresh_history_files, None, [history_dropdown])
    load_btn.upload(load_history, load_btn, [chatbot])
    refresh_btn.click(refresh_history_files, None, [history_dropdown]).then(load_history_from_dropdown, history_dropdown, [chatbot])
    history_dropdown.change(load_history_from_dropdown, history_dropdown, [chatbot])
    open_folder_btn.click(open_history_folder, None, None)
    temperature.change(update_config, [temperature, top_k, top_p, repeat_penalty, seed], None)
    top_k.change(update_config, [temperature, top_k, top_p, repeat_penalty, seed], None)
    top_p.change(update_config, [temperature, top_k, top_p, repeat_penalty, seed], None)
    repeat_penalty.change(update_config, [temperature, top_k, top_p, repeat_penalty, seed], None)
    seed.change(update_config, [temperature, top_k, top_p, repeat_penalty, seed], None)
    random_seed_btn.click(random_seed, None, [seed])

    # 页面加载时清除历史
    demo.load(model_history_restart, None, [chatbot])


# 入口函数
if __name__ == '__main__':
    atexit.register(stop_at_exit)  # 注册退出
    # 自动打开浏览器
    webbrowser.open_new_tab(f"http://{configs['host']}:{configs['uiport']}")  # 打开默认浏览器的新标签页
    # 启动，运行在0.0.0.0本机上，可以通过修改系统环境变量GRADIO_SERVER_NAME来改变
    demo.launch(server_name='0.0.0.0', server_port=int(configs['uiport']))
