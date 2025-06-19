from dotenv import load_dotenv
from typing import List, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import MessageGraph, END

from chains import generate_chain, reflect_chain

load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    """Generate a response based on the input messages."""
    response = generate_chain.invoke({"messages": messages})
    return response

def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    """Reflect on the input messages and provide critique."""
    response = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=response.content)]

def should_continue(messages: Sequence[BaseMessage]) -> str:
    return END if len(messages) > 6 else REFLECT

def create_reflection_graph() -> MessageGraph:
    builder = MessageGraph()

    builder.add_node(GENERATE, generation_node)
    builder.add_node(REFLECT, reflection_node)

    builder.set_entry_point(GENERATE)

    builder.add_conditional_edges(GENERATE, should_continue, {
    REFLECT: REFLECT,
    END: END
})
    builder.add_edge(REFLECT, GENERATE)

    return builder

if __name__ == "__main__":
    builder = create_reflection_graph()
    graph = builder.compile()

    print(graph.get_graph().draw_mermaid())
    graph.get_graph().draw_mermaid_png(output_file_path="flow.png")
    graph.get_graph().print_ascii()

    inputs = [
        HumanMessage(content="""Make this tweet better:
@LangChainAI — newly Tool Calling feature is seriously underrated.

After a long wait, it's here — making the implementation of agents across different models with function calling super easy.

Made a video covering their newest blog post.""")
    ]
    response = graph.invoke(inputs)
    print("\nFinal Output:")
    for msg in response:
        print(f"{msg.type}: {msg.content}")
