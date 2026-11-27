from agent_obs.versioning.extractor import AgentConfigurationExtractor


class MyAgent:
    system_prompt = "You are a helpful SQL assistant."
    examples = []
    tools = []
    model = "GPT-5"
    temperature = 0.2
    top_p = 1.0
    response_schema = None
    safety_instructions = None
    memory_template = None
    retrieval_template = None


agent = MyAgent()

artifact = AgentConfigurationExtractor.from_agent(agent)

print(artifact)
