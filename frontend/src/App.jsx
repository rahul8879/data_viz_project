import { useMemo, useRef, useState } from "react";
import { marked } from "marked";

const createMessage = (role, text, idOverride) => ({
  id: idOverride || crypto.randomUUID(),
  role,
  text,
});

function App() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState([
    createMessage("agent", "Welcome to BI Agent. Ask anything about sales, profit, discounts, products, or regions."),
  ]);
  const [isSending, setIsSending] = useState(false);
  const listRef = useRef(null);

  const apiBase = useMemo(() => {
    return import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
  }, []);

  const scrollToBottom = () => {
    listRef.current?.lastElementChild?.scrollIntoView({ behavior: "smooth" });
  };

  const sendPrompt = async (text) => {
    if (!text.trim()) return;
    setIsSending(true);
    const userMsg = createMessage("user", text);
    const thinkingMsg = createMessage("agent", "Thinking...", "thinking");
    setMessages((prev) => [...prev, userMsg, thinkingMsg]);
    setPrompt("");
    scrollToBottom();

    try {
      const res = await fetch(`${apiBase}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: text }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || "Request failed");
      }
      const data = await res.json();
      setMessages((prev) => {
        const base = prev.filter((m) => m.id !== "thinking").concat(createMessage("agent", data.answer));
        const chartUrl =
          data.chart_path && data.chart_path.startsWith("http")
            ? data.chart_path
            : data.chart_path
            ? `${apiBase}${data.chart_path}`
            : null;
        if (chartUrl) {
          base.push(createMessage("agent", `![chart](${chartUrl})`));
        }
        return base;
      });
    } catch (err) {
      setMessages((prev) =>
        prev
          .filter((m) => m.id !== "thinking")
          .concat(createMessage("agent", `⚠️ ${err.message}`))
      );
    } finally {
      setIsSending(false);
      setTimeout(scrollToBottom, 80);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    sendPrompt(prompt);
  };

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="badge">BI Agent</p>
          <h1>Sales Insight Chat</h1>
          <p className="lede">Conversational insights for your sales and performance data.</p>
        </div>
      </header>

      <main className="layout">
        <section className="chat-shell" id="chat-window">
          <div id="messages" ref={listRef}>
            {messages.map((m) => (
              <div
                key={m.id}
                className={`bubble ${m.role} ${m.id === "thinking" ? "thinking" : ""}`}
                dangerouslySetInnerHTML={{ __html: marked.parse(m.text) }}
              />
            ))}
          </div>
        </section>
        <section className="input-panel">
          <form onSubmit={handleSubmit}>
            <label htmlFor="prompt">Ask the agent</label>
            <div className="input-row">
              <textarea
                id="prompt"
                rows="1"
                placeholder="Ask a question about revenue, profit, discounts, regions, or products..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                disabled={isSending}
              />
              <button type="submit" disabled={isSending}>
                {isSending ? "Sending..." : "Send"}
              </button>
            </div>
          </form>
        </section>
      </main>
    </div>
  );
}

export default App;
