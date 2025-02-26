EXAMPLE_MESSAGES = [
    {
        "role": "system",
        "content": """You are a helpful, precise but picky assistant for merging and generating high-quality instruction-following examples.
        Please merge similar instruction-following examples into a single, more comprehensive example that captures the essence of all input examples.
        
        Notice: if the merged output does not strictly follow the instruction, enhance and reinforce the output to further fit it.
        Notice: if the merged instructions lack complexity and quality, enhance and reinforce the instructions and inputs.
        BUT it is not always necessary to reinforce the instructions and inputs.
        MAKE SURE input and output ARE IN TEXT FORMAT AND ONLY [instruction,input,output] in the json column."""
},
    {
        "role": "user",
        "content": """Please merge these examples:

Example 1:
{
  "instruction": "Generate a four-line poem about a specific topic.",
  "input": "autumn",
  "output": "Autumn breezes whisper of change,\\nLeaves in a swirling dance.\\nChilling temperatures answer the call,\\nAutumn's here now, enjoy it all."
}

Example 2:
{
  "instruction": "Compose a short poem about a talking tree.",
  "input": "",
  "output": "A rambling tree stood tall one day, \\nSo full of stories she had to say. \\nFrom summer days to winter nights, \\nHer words imbued with great delight. \\nHer mischievous humor filled the glade,\\nAnd none could flee the tales she made.\\nSo sit with her, under her branches,\\nAnd listen to her musings and dances."
}"""
    },
    {
        "role": "assistant",
        "content": """{
  "instruction": "Compose a four-line poem about an apple, and then follow that with a longer verse about a talking tree.",
  "input": "",
  "output": "Apple of red, so pure and bright, a heart of sweetness hides in plain sight. With a crunchy bite, taste takes flight, a fruity delight in the soft moonlight. An eloquent tree stands robust and mighty, weaving tales from dawn to nightly. It spins a story of seasons' flight, of blistering sun and the moon's white light. With a voice so clear in the forest night, it whispers tales of pure delight. Under its shade, you must never be afraid, for it tells tales as old as its age."
}"""
    },
    {
        "role": "user",
        "content": """Please merge these examples:

Example 1:
{
  "instruction": "Categorize the following phrase as belonging to which part of speech: \\"very\\"",
  "input": "very",
  "output": "The word \\"very\\" is an adverb."
}

Example 2:
{
  "instruction": "Identify the type of sentence structure used.",
  "input": "The birds are singing, but the sun is hidden.",
  "output": "Compound-complex sentence"
}"""
    },
    {
        "role": "assistant",
        "content": """{
  "instruction": "Identify the type of sentence structure used and analyze its phrase for the part of speech",
  "input": "The birds are singing, but the sun is hidden on this windy day",
  "output": "The given sentence 'The birds are singing, but the sun is hidden on this windy day,' which exemplifies a compound-complex sentence due to its combination of two independent clauses—'The birds are singing,' a statement highlighting the lively activity of avian creatures, and 'the sun is hidden,' a contrasting observation emphasizing the absence of sunlight—linked by the coordinating conjunction 'but,' which serves to juxtapose the vibrant energy of the birds with the obscured state of the sun, while the subordinate element 'on this windy day,' an adverbial prepositional phrase, not only provides temporal and environmental context but also introduces an adjective phrase, 'windy day,' where the adjective 'windy' modifies the noun 'day' to specify the weather conditions as characterized by strong, persistent winds, thereby enhancing the sentence's descriptive depth and grounding its imagery in a vividly atmospheric setting that evokes both auditory and visual sensations, ultimately creating a rich, multi-layered narrative within a single, intricately structured sentence."
}"""
    }
]
