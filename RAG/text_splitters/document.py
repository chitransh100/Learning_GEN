from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.GO,
    chunk_size=100,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(f"Number of chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}:")
    print(chunk)