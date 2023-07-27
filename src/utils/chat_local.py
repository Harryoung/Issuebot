import argparse
import os
from pydoc import describe
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import SimpleSequentialChain
from langchain import PromptTemplate

import openai

os.environ["OPENAI_API_KEY"] = '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'
os.environ ["ACTIVELOOP_USERNAME"] ='shiyutang'
os.environ ["ACTIVELOOP_TOKEN"] ='eyJhbGciOiJIUzUxMiIsImlhdCI6MTY5MDE2Nzk4MiwiZXhwIjoxNzIxNzkwMzAwfQ.eyJpZCI6InNoaXl1dGFuZyJ9.OW67VVUzR_0Vhk2qhp1pgF0gD4R_mjtTfqJaVz28i-K8p__6vINDWPoWkxjSqRkHL4w61Z-dQv3csIvvgJEa1Q'

class Chatbot():
    def __init__(self, args):
        # Set the OpenAI API key from the environment variable
        openai.api_key = os.environ.get("OPENAI_API_KEY")

        # Create an instance of OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()

        # Create an instance of DeepLake with the specified dataset path and embeddings
        self.db = DeepLake(
            dataset_path=args.activeloop_dataset_path,
            read_only=True,
            embedding_function=embeddings,
        )
        # Create a ChatOpenAI model instance
        self.model = ChatOpenAI(model="gpt-3.5-turbo")

        self.init_retriever()
    
    def init_retriever(self):
        # Create a retriever from the DeepLake instance

        retriever = self.db.as_retriever()
        # Set the search parameters for the retriever
        retriever.search_kwargs["distance_metric"] = "cos"
        retriever.search_kwargs["fetch_k"] = 100
        retriever.search_kwargs["maximal_marginal_relevance"] = True
        retriever.search_kwargs["k"] = 10

        self.retriever = retriever

    def answer_query(self, query):
        """Search for a response to the query in the DeepLake database."""
        
        # summarise_chain = load_summarize_chain(self.model, chain_type="map_reduce") # How to not summarize
        
        # Create a RetrievalQA instance from the model and retriever
        # qa = RetrievalQA.from_llm(model, retriever=retriever)

        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Use the language used in the question to answer the questions please.  

        {context}
        Question: {question}

        Helpful Answer:"""
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

        qa_chain = RetrievalQA.from_chain_type(llm=self.model, chain_type="map_reduce", retriever=self.retriever, chain_type_kwargs={"return_intermediate_steps": False,}) # "prompt": QA_CHAIN_PROMPT

        overall_chain = SimpleSequentialChain(chains=[qa_chain], verbose=True)
        import pdb; pdb.set_trace()
        
        return overall_chain.run(query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activeloop_dataset_path", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    chatbot = Chatbot(args)
    print("ready to find some answers")
    chatbot.answer_query(args.query)



# testcases:
# Let's say I want to train an Arabic recognition model, what's the best practice when creating a customized Arabic dictionary?\nNow, there are several things that make it challenging:\nArabic letters change their shapes depending on their locations in the word, for example, the letter alif has 4 forms and each one has a unicode glyph. Should I include all possible shapes of it in the dictionary or should I just include a single letter in the alphabet?\nfollow-up on 1, if I only include a single letter, then how is the model trained such that it can recognize different shapes of the same letter? It sounds like a 1-to-many mapping, can the model do that?\nArabic is cursive, that means when joining letters together, they merge together, which is called ligature. How can I take this into account when creating the dictionary?\nWhat's the order of paddleOCR recognition? Because Arabic is a right-to-left language, and if paddleocr reads texts from left to right, should I be concerned and are there any files that I should change?
# 测试公式识别CAN算法。使用的命令:python3 tools/infer_rec.py -c configs/rec/rec_d28_can.yml -o Architecture.Head.attdecoder.is_train=False Global.infer_img='/data2/ocr/mathematicalRec/gray' Global.pretrained_model=./weights/rec_d28_can_train/best_accuracy.pdparams\ntext_index[batch_idx] Tensor(shape=[36], dtype=int64, place=CUDAPlace(0), stop_gradient=True,[86, 86, 52, 107, 12, 6 , 86, 49, 49, 86, 49, 86, 49, 86, 12, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ])\nseq_end is Tensor(shape=[], dtype=int64, place=CUDAPlace(0), stop_gradient=True,15)\nhttps://github.com/PaddlePaddle/PaddleOCR/blob/fac03876f39bc67acc8eef6d7facb9a2206eeecd/ppocr/postprocess/rec_postprocess.py#L913C54-L913C54 这里应该使用：\nseq_end = text_index[batch_idx].argmin()\n如果出现最小值在0号位或text_index[batch_idx]全0即(text_index[batch_idx] == 0).all() 为 True，如text_index[batch_idx] Tensor(shape=[36], dtype=int64, place=CUDAPlace(0), stop_gradient=True, [4 , 78, 8 , 97, 5 , 6 , 12, 12, 8 , 12, 6 , 12, 22, 12, 22, 78, 91, 22, 78, 8 , 12, 8 , 12, 22, 12, 22, 12, 22, 12, 22, 12, 22, 12, 22, 12, 22])\n则在idx_list = text_index[batch_idx][:seq_end].tolist()会出现\nValueError: (InvalidArgument) Attr(ends) should be greater than attr(starts) in slice op. But received end = 0, start = 0.测试公式识别CAN算法。使用的命令:python3 tools/infer_rec.py -c configs/rec/rec_d28_can.yml -o Architecture.Head.attdecoder.is_train=False Global.infer_img='/data2/ocr/mathematicalRec/gray' Global.pretrained_model=./weights/rec_d28_can_train/best_accuracy.pdparams\ntext_index[batch_idx] Tensor(shape=[36], dtype=int64, place=CUDAPlace(0), stop_gradient=True,[86, 86, 52, 107, 12, 6 , 86, 49, 49, 86, 49, 86, 49, 86, 12, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ])\nseq_end is Tensor(shape=[], dtype=int64, place=CUDAPlace(0), stop_gradient=True,15)\nhttps://github.com/PaddlePaddle/PaddleOCR/blob/fac03876f39bc67acc8eef6d7facb9a2206eeecd/ppocr/postprocess/rec_postprocess.py#L913C54-L913C54 这里应该使用：\nseq_end = text_index[batch_idx].argmin()\n如果出现最小值在0号位或text_index[batch_idx]全0即(text_index[batch_idx] == 0).all() 为 True，如text_index[batch_idx] Tensor(shape=[36], dtype=int64, place=CUDAPlace(0), stop_gradient=True, [4 , 78, 8 , 97, 5 , 6 , 12, 12, 8 , 12, 6 , 12, 22, 12, 22, 78, 91, 22, 78, 8 , 12, 8 , 12, 22, 12, 22, 12, 22, 12, 22, 12, 22, 12, 22, 12, 22])\n则在idx_list = text_index[batch_idx][:seq_end].tolist()会出现\nValueError: (InvalidArgument) Attr(ends) should be greater than attr(starts) in slice op. But received end = 0, start = 0.测试公式识别CAN算法。使用的命令:python3 tools/infer_rec.py -c configs/rec/rec_d28_can.yml -o Architecture.Head.attdecoder.is_train=False Global.infer_img='/data2/ocr/mathematicalRec/gray' Global.pretrained_model=./weights/rec_d28_can_train/best_accuracy.pdparams\ntext_index[batch_idx] Tensor(shape=[36], dtype=int64, place=CUDAPlace(0), stop_gradient=True,[86, 86, 52, 107, 12, 6 , 86, 49, 49, 86, 49, 86, 49, 86, 12, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ])\nseq_end is Tensor(shape=[], dtype=int64, place=CUDAPlace(0), stop_gradient=True,15)\nhttps://github.com/PaddlePaddle/PaddleOCR/blob/fac03876f39bc67acc8eef6d7facb9a2206eeecd/ppocr/postprocess/rec_postprocess.py#L913C54-L913C54 这里应该使用：\nseq_end = text_index[batch_idx].argmin()\n如果出现最小值在0号位或text_index[batch_idx]全0即(text_index[batch_idx] == 0).all() 为 True，如text_index[batch_idx] Tensor(shape=[36], dtype=int64, place=CUDAPlace(0), stop_gradient=True, [4 , 78, 8 , 97, 5 , 6 , 12, 12, 8 , 12, 6 , 12, 22, 12, 22, 78, 91, 22, 78, 8 , 12, 8 , 12, 22, 12, 22, 12, 22, 12, 22, 12, 22, 12, 22, 12, 22])\n则在idx_list = text_index[batch_idx][:seq_end].tolist()会出现\nValueError: (InvalidArgument) Attr(ends) should be greater than attr(starts) in slice op. But received end = 0, start = 0.
# Hey, when building an exe with Pyinstaller from a python file that uses PaddleOCR I get a \"No module named 'ppocr' error\" (attached a screenshot). How do I fix it? \nThe code I'm using is just the most barebones one to find out how to create an exe file with PaddleOCR:\n`import paddleocr\nocr = paddleocr.PaddleOCR()\npath = input('input path: ')\nresult = ocr.ocr(path)\nprint(result)
