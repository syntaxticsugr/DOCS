import gradio as gr
from rag import langchain_rag_search



with gr.Blocks(title="DOCS") as app:

    with gr.Row():

        gr.HTML("""
            <div>
                <h1>DOCS: LangChain RAG Demo</h1>
                <h3>DOCS is a simple LangChain RAG utility to search within and read docs of your favourite libraries easily.<h3>
            </div>
        """)

        with gr.Column():

            apikey_openai_input = gr.Textbox(label="OpenAI API Key:", interactive=True, type='password')

            gr.Markdown("[//]: # (Only Here To Add Blank Space)")

            docs_link_input = gr.Textbox(label="Link To Docs:", interactive=True, max_lines=1)

    gr.Markdown("[//]: # (Only Here To Add Blank Space)")

    with gr.Row():

        search_query_input = gr.TextArea(label="Search:", interactive=True, scale=3, lines=2, max_lines=6)

        search_query_submit_button = gr.Button("Submit", variant='primary', scale=1)
    
    gr.Markdown("[//]: # (Only Here To Add Blank Space)")

    output = gr.TextArea(label="Result:", interactive=False)

    search_query_submit_button.click(fn=langchain_rag_search, inputs=[apikey_openai_input, docs_link_input, search_query_input], outputs=output)



app.launch()
