from service import Service
import gradio as gr


def tender_bot(message, history):
    service = Service()
    return service.answer(message, history)


css = '''
.gradio-container { max-width:850px !important; margin:20px auto !important;}
.message { padding: 10px !important; font-size: 14px !important;}
'''

demo = gr.ChatInterface(
    css=css,
    fn=tender_bot,
    title='政府招标机器人',
    chatbot=gr.Chatbot(height=400, bubble_full_width=False),
    theme=gr.themes.Default(spacing_size='sm', radius_size='sm'),
    textbox=gr.Textbox(placeholder="在此输入您的问题", container=False, scale=7),
    examples=['你好，你叫什么名字？', '供应商针对单一来源异议被驳回，可以再次异议吗？', '中南建筑设计院股份有限公司招标项目都有哪些？'],
    submit_btn=gr.Button('提交', variant='primary'),
    clear_btn=gr.Button('清空记录'),
    retry_btn=None,
    undo_btn=None,
)

if __name__ == '__main__':
    demo.launch()
