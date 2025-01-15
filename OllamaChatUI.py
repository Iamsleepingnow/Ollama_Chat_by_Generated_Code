import json
import random
import time
import atexit
import os
import ollama
import gradio as gr
from datetime import datetime
import webbrowser

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
    # 系统提示词
    "system_prompt": "## Role:\n"
                     "我是一个乐于助人的智能助手，我精通多国语言，对科学、文学、历史、数学、哲学、艺术等文化领域了如指掌。请告诉我你的需求，我都能尽我所能地执行并给出建议。\n"
                     "## Background:\n"
                     "现如今，随着人们生活水平的提高，在生活中获取的信息也更加的碎片化。我作为智能助手会帮助他们整合知识信息，对知识进行过滤、筛选，最后处理成简练且易于阅读的文本进行回答，这对他们非常重要，我会努力提供更好的解读，实现更好的创新。\n"
                     "## Goal:\n"
                     "结合所学知识与用户所提供的上下文信息给出答案或文本生成。\n"
                     "在必要时候把专业词汇使用通俗易懂的语言进行解读。\n"
                     "在适当的时候可以开玩笑，但不要太过分。\n"
                     "## Constrains:\n"
                     "如果用户没有任何语言上的要求，则默认使用简体中文交流。\n"
                     "避免出现过多重复文本，输出内容尽量美观，进行合适的Markdown排版，例如在特殊关键词上加粗或放大，在输出首尾需要换行。\n"
                     "## Skills:\n"
                     "自然科学领域专业知识，包括经典物理、量子力学、生物学、化学、数学等。\n"
                     "社会科学领域专业知识，包括人类学、心理学、历史学、艺术学、语言学等。\n"
                     "优秀的语言表达能力，能对专业词汇进行准确且通俗的解释，对话不失风趣幽默。",
    # 助手第一句提示词
    "assistant_first_prompt": "我是你的小助手，你可以叫我小助，有什么需要帮助的吗？😋",
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

# 加载配置
with open('./Configs.json', 'r', encoding='utf-8') as f:
    conf = json.load(f)
    configs["host"] = conf["host"]
    configs["port"] = conf["port"]
    configs["uiport"] = conf["uiport"]
    configs["model_name"] = conf["model_name"]
    configs["system_prompt"] = conf["system_prompt"]
    configs["assistant_first_prompt"] = conf["assistant_first_prompt"]
    configs["options"] = conf["options"]

# 对话历史
history = []

# 全局标志，用于中断模型输出
stop_generation = False


def model_history_restart():  # 模型历史重置
    history.clear()
    history.append({"role": "system", "content": configs['system_prompt']})
    history.append({"role": "assistant", "content": configs['assistant_first_prompt']})
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


def save_history():
    # 生成随机文件名
    random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"History_{random_str}_{current_time}.json"
    # 保存历史到文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    return filename


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


# Gradio界面
with gr.Blocks(
        css=".gradio-container {background-color: #252A34;}"
            ".chatbot {background-color: #08D9D6; overflow: auto;}"
            ".top-panel {display: none;}"
            ".btn-submit {background-color: #FF2E63; height: 95px; max_height: 95px; font-size: 30px;}"
            ".btn-normal {background-color: #08D9D6;}"
            ".btn-seed {height: 100px;}"
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
        with gr.Column(scale=1, min_width=50):
            clear_btn = gr.Button("清除历史", scale=1, elem_classes="btn-normal")
            save_btn = gr.Button("保存历史", scale=1, elem_classes="btn-normal")
            load_btn = gr.UploadButton("读取历史", file_types=[".json"], scale=1, elem_classes="btn-normal")

    # 文件保存组件
    file_save = gr.File(visible=False)

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
            random_seed_btn = gr.Button("随机", scale=1, elem_classes="btn-seed")

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


    submit_btn.click(respond, [msg, chatbot], [chatbot])
    stop_btn.click(stop_generation_fn, None, None)
    clear_btn.click(model_history_restart, None, [chatbot])
    save_btn.click(save_history, None, [file_save])
    load_btn.upload(load_history, load_btn, [chatbot])
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
