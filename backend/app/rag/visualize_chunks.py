import os
import re
import yaml
import json
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

def custom_markdown_reader(file_path: str):
    """Loads markdown and parses frontmatter metadata."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    
    metadata = {}
    text_content = content
    
    if frontmatter_match:
        frontmatter_text = frontmatter_match.group(1)
        try:
            metadata = yaml.safe_load(frontmatter_text) or {}
        except Exception:
            pass
        text_content = content[frontmatter_match.end():]
        
    clean_metadata = {}
    for k, v in metadata.items():
        clean_metadata[k.strip().lower()] = str(v).strip()

    clean_metadata["file_name"] = os.path.basename(file_path)
    
    for key in ["category", "distance_tier", "experience_level", "topic", "source"]:
        if key not in clean_metadata:
            clean_metadata[key] = "all"

    return Document(text=text_content, metadata=clean_metadata)  # type: ignore

def build_visualizer():
    print("=" * 60)
    print("🎨 GENERATING CUSTOM CHUNK VISUALIZER")
    print("=" * 60)
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    if not os.path.exists(data_dir):
        print(f"❌ Data directory does not exist: {data_dir}")
        return
        
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".md"):
            file_path = os.path.join(data_dir, filename)
            documents.append(custom_markdown_reader(file_path))
            
    # Chunk them
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"Loaded {len(documents)} documents, split into {len(nodes)} chunks.")
    
    # Prepare serializable list of chunks for JS
    chunks_data = []
    for idx, node in enumerate(nodes):
        chunks_data.append({
            "id": node.node_id,
            "index": idx + 1,
            "file_name": node.metadata.get("file_name"),
            "category": node.metadata.get("category", "unknown"),
            "topic": node.metadata.get("topic", "general").replace("_", " ").title(),
            "experience_level": node.metadata.get("experience_level", "all"),
            "distance_tier": node.metadata.get("distance_tier", "all"),
            "source": node.metadata.get("source", "unknown"),
            "length": len(node.text),  # type: ignore
            "text": node.text  # type: ignore
        })
        
    # Generate HTML content
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LlamaIndex Chunk Ingestion Viewer</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #090b11;
            --surface: #131722;
            --surface-hover: #1e2435;
            --primary: #4f46e5;
            --primary-light: #818cf8;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --border: #272d3d;
            --tag-bg: #1e1b4b;
            --tag-text: #c7d2fe;
        }}
        
        body {{
            margin: 0;
            padding: 0;
            background-color: var(--bg);
            color: var(--text-main);
            font-family: 'Plus Jakarta Sans', sans-serif;
            min-height: 100vh;
        }}
        
        header {{
            background: linear-gradient(135deg, #131722 0%, #090b11 100%);
            padding: 2.5rem 2rem;
            border-bottom: 1px solid var(--border);
            text-align: center;
        }}
        
        h1 {{
            margin: 0;
            font-weight: 800;
            font-size: 2.2rem;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .subtitle {{
            color: var(--text-muted);
            margin-top: 0.5rem;
            font-size: 1rem;
        }}

        .stats-container {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }}

        .stat-card {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            min-width: 120px;
            text-align: center;
        }}

        .stat-value {{
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--primary-light);
        }}

        .stat-label {{
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}
        
        .main-layout {{
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 1.5rem;
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 2rem;
        }}

        @media (max-width: 900px) {{
            .main-layout {{
                grid-template-columns: 1fr;
            }}
        }}

        .controls-pane {{
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            height: fit-content;
            position: sticky;
            top: 2rem;
        }}

        .search-box input {{
            width: 100%;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            background: var(--bg);
            border: 1px solid var(--border);
            color: var(--text-main);
            box-sizing: border-box;
            font-family: inherit;
            margin-bottom: 1.5rem;
        }}

        .search-box input:focus {{
            border-color: var(--primary-light);
            outline: none;
        }}

        .filter-section {{
            margin-bottom: 1.5rem;
        }}

        .filter-title {{
            font-weight: 600;
            font-size: 0.9rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.75rem;
        }}

        .btn-group {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        .filter-btn {{
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 0.6rem 1rem;
            border-radius: 8px;
            text-align: left;
            font-family: inherit;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .filter-btn:hover, .filter-btn.active {{
            background: var(--primary);
            color: white;
            border-color: var(--primary-light);
        }}
        
        .chunks-list {{
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }}
        
        .chunk-card {{
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        
        .chunk-card:hover {{
            transform: translateY(-2px);
            border-color: var(--border-hover, #4f46e5);
            background-color: var(--surface-hover);
        }}
        
        .chunk-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.75rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        
        .chunk-title-group {{
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}
        
        .chunk-index {{
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--primary-light);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .chunk-topic {{
            font-size: 1.2rem;
            font-weight: 600;
            margin: 0;
            color: var(--text-main);
        }}
        
        .badges-group {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}
        
        .badge {{
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .badge-category {{ background-color: #1e1b4b; color: #c7d2fe; }}
        .badge-experience {{ background-color: #064e3b; color: #a7f3d0; }}
        .badge-distance {{ background-color: #701a75; color: #fbcfe8; }}
        
        .chunk-text {{
            font-size: 0.95rem;
            line-height: 1.6;
            color: #d1d5db;
            white-space: pre-wrap;
            background: rgba(0,0,0,0.2);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.03);
        }}
        
        .chunk-footer {{
            margin-top: 1rem;
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--text-muted);
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 0.75rem;
        }}

        .empty-state {{
            text-align: center;
            padding: 4rem 2rem;
            background: var(--surface);
            border: 1px dashed var(--border);
            border-radius: 16px;
            color: var(--text-muted);
        }}
    </style>
</head>
<body>

    <header>
        <h1>LlamaIndex Chunk Ingestion Viewer</h1>
        <div class="subtitle">Observe how SentenceSplitter parsed and prepared your Markdown knowledge base chunks</div>
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-value">{len(documents)}</div>
                <div class="stat-label">Documents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(nodes)}</div>
                <div class="stat-label">Total Chunks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{int(sum(len(n.get_content()) for n in nodes)/len(nodes)) if nodes else 0}</div>
                <div class="stat-label">Avg Chunk Size (chars)</div>
            </div>
        </div>
    </header>

    <div class="main-layout">
        <div class="controls-pane">
            <div class="search-box">
                <div class="filter-title">Search Text</div>
                <input type="text" id="searchInput" placeholder="Search keywords..." oninput="filterChunks()">
            </div>
            
            <div class="filter-section">
                <div class="filter-title">Category</div>
                <div class="btn-group">
                    <button class="filter-btn active" onclick="setCategoryFilter('all', this)">All Categories</button>
                    <button class="filter-btn" onclick="setCategoryFilter('nutrition', this)">Nutrition</button>
                    <button class="filter-btn" onclick="setCategoryFilter('mobility', this)">Mobility</button>
                    <button class="filter-btn" onclick="setCategoryFilter('training', this)">Training</button>
                    <button class="filter-btn" onclick="setCategoryFilter('medical', this)">Medical</button>
                </div>
            </div>
        </div>

        <div>
            <div class="chunks-list" id="chunksContainer">
                <!-- Chunks will be injected here by Javascript -->
            </div>
        </div>
    </div>

    <script>
        const chunks = {json.dumps(chunks_data)};
        let activeCategory = 'all';

        function renderChunks(filteredList) {{
            const container = document.getElementById("chunksContainer");
            container.innerHTML = "";
            
            if (filteredList.length === 0) {{
                container.innerHTML = `
                    <div class="empty-state">
                        <h3>No matching chunks found</h3>
                        <p>Try adjusting your search keywords or category filters.</p>
                    </div>
                `;
                return;
            }}

            filteredList.forEach(chunk => {{
                const card = document.createElement("div");
                card.className = "chunk-card";
                card.innerHTML = `
                    <div class="chunk-header">
                        <div class="chunk-title-group">
                            <span class="chunk-index">Chunk #${{chunk.index}} (ID: ${{chunk.id.substring(0, 8)}}...)</span>
                            <h3 class="chunk-topic">${{chunk.topic}}</h3>
                        </div>
                        <div class="badges-group">
                            <span class="badge badge-category">${{chunk.category}}</span>
                            <span class="badge badge-experience">${{chunk.experience_level}}</span>
                            <span class="badge badge-distance">${{chunk.distance_tier}}</span>
                        </div>
                    </div>
                    <div class="chunk-text">${{escapeHtml(chunk.text)}}</div>
                    <div class="chunk-footer">
                        <span>Source: <strong>${{chunk.source}}</strong> (${{chunk.file_name}})</span>
                        <span>Length: ${{chunk.length}} chars</span>
                    </div>
                `;
                container.appendChild(card);
            }});
        }}

        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.innerText = text;
            return div.innerHTML;
        }}

        function setCategoryFilter(category, btnElement) {{
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            btnElement.classList.add('active');
            activeCategory = category;
            filterChunks();
        }}

        function filterChunks() {{
            const searchVal = document.getElementById("searchInput").value.toLowerCase();
            
            const filtered = chunks.filter(chunk => {{
                const matchesCategory = activeCategory === 'all' || chunk.category === activeCategory;
                const matchesSearch = chunk.text.toLowerCase().includes(searchVal) || 
                                      chunk.topic.toLowerCase().includes(searchVal) ||
                                      chunk.file_name.toLowerCase().includes(searchVal);
                return matchesCategory && matchesSearch;
            }});
            
            renderChunks(filtered);
        }}

        // Initial render
        renderChunks(chunks);
    </script>
</body>
</html>
"""
    
    # Save the output HTML file to root backend folder
    viewer_path = os.path.join(os.path.dirname(__file__), "..", "..", "chunk_viewer.html")
    with open(viewer_path, "w", encoding="utf-8") as f:
        f.write(html_template)
        
    print("\n" + "=" * 60)
    print("✅ VISUALIZER GENERATED SUCCESSFULLY!")
    print(f"Saved to: {os.path.abspath(viewer_path)}")
    print("Double-click this file to open it in your browser!")
    print("=" * 60)

if __name__ == "__main__":
    build_visualizer()
