import "dotenv/config";
import { MultiRetrievalQAChain } from "langchain/chains";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { OpenAIChat } from "langchain/llms/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import readline from "readline";

(async () => {
  const vectorStore = await MemoryVectorStore.fromTexts(
    [
      "Tears of the Kingdom takes place a number of years after Breath of the Wild, at the end of the Zelda timeline in the kingdom of Hyrule. Link and Zelda set out to explore a cavern beneath Hyrule Castle, from which gloom, a poisonous substance, has been seeping out and causing people to fall ill.",
    ],
    { series: "Zelda" },
    new OpenAIEmbeddings()
  );

  const llm = new OpenAIChat({});

  const chain = MultiRetrievalQAChain.fromLLMAndRetrievers(llm, {
    retrieverNames: ["zelda"],
    retrieverDescriptions: [
      "Good for answering questions about Zelda Tears of the kingdom",
    ],
    retrievers: [vectorStore.asRetriever()],
    retrievalQAChainOpts: {
      returnSourceDocuments: true,
    },
  });

  const showCliPrompt = async () => {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    rl.question(
      "Ask anything about Zelda TOTK\n eg: Tears of the Kingdom, what is it? \n",
      async (question) => {
        const res = await chain.call({ input: question });
        if (res.result) {
          console.log(res.result);
        } else {
          console.log(res.text);
          console.log(res.sourceDocuments);
        }
        console.log("");
        rl.close();
        showCliPrompt();
      }
    );
  };
  showCliPrompt();
})();
