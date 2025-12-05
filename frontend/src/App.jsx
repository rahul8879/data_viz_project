import { useEffect, useMemo, useRef, useState } from "react";
import { marked } from "marked";
import * as sdk from "microsoft-cognitiveservices-speech-sdk";

const createMessage = (role, text, idOverride) => ({
  id: idOverride || crypto.randomUUID(),
  role,
  text,
});

const quickPrompts = [
  "Top regions by revenue last month",
  "Which products saw the biggest discount?",
  "How is profit trending week over week?",
  "Show me sales by channel and segment",
];

function App() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState([
    createMessage(
      "agent",
      "Hi, I'm your Sales Insight Agent. Ask me about revenue, profit, discounts, products, or regions—I’ll do the heavy lifting and summarize the answer."
    ),
  ]);
  const [isSending, setIsSending] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [speechError, setSpeechError] = useState("");
  const listRef = useRef(null);
  const speechRecognizerRef = useRef(null);

  const apiBase = useMemo(() => {
    return import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
  }, []);

  const speechConfig = useMemo(
    () => ({
      key: import.meta.env.VITE_AZURE_SPEECH_KEY,
      region: import.meta.env.VITE_AZURE_SPEECH_REGION,
      language: import.meta.env.VITE_AZURE_SPEECH_LANG || "en-US",
    }),
    []
  );

  const scrollToBottom = () => {
    listRef.current?.lastElementChild?.scrollIntoView({ behavior: "smooth" });
  };

  const cleanupRecognizer = () => {
    speechRecognizerRef.current?.close();
    speechRecognizerRef.current = null;
  };

  const stopListening = (silent = false) => {
    const recognizer = speechRecognizerRef.current;
    if (!recognizer) {
      setIsRecording(false);
      return;
    }
    recognizer.stopContinuousRecognitionAsync(
      () => {
        cleanupRecognizer();
        setIsRecording(false);
      },
      (err) => {
        cleanupRecognizer();
        setIsRecording(false);
        if (!silent) {
          setSpeechError(err?.message || "Could not stop speech recognition.");
        }
      }
    );
  };

  const startListening = () => {
    if (!speechConfig.key || !speechConfig.region) {
      setSpeechError("Add VITE_AZURE_SPEECH_KEY and VITE_AZURE_SPEECH_REGION to use the microphone.");
      return;
    }
    setSpeechError("");
    try {
      const azureSpeechConfig = sdk.SpeechConfig.fromSubscription(speechConfig.key, speechConfig.region);
      azureSpeechConfig.speechRecognitionLanguage = speechConfig.language;
      const audioConfig = sdk.AudioConfig.fromDefaultMicrophoneInput();
      const recognizer = new sdk.SpeechRecognizer(azureSpeechConfig, audioConfig);
      speechRecognizerRef.current = recognizer;

      recognizer.recognized = (_, e) => {
        if (e.result?.reason === sdk.ResultReason.RecognizedSpeech && e.result.text) {
          setPrompt((prev) => (prev ? `${prev.trim()} ${e.result.text}` : e.result.text));
        }
      };

      recognizer.canceled = (_, e) => {
        stopListening(true);
        if (e.errorDetails) {
          setSpeechError(e.errorDetails);
        }
      };

      recognizer.sessionStopped = () => {
        stopListening(true);
      };

      recognizer.startContinuousRecognitionAsync(
        () => setIsRecording(true),
        (err) => {
          setSpeechError(err?.message || "Could not start speech recognition.");
          stopListening(true);
        }
      );
    } catch (err) {
      setSpeechError(err.message || "Speech setup failed.");
      stopListening(true);
    }
  };

  const toggleListening = () => {
    if (isRecording) {
      stopListening();
    } else {
      startListening();
    }
  };

  const sendPrompt = async (text) => {
    if (!text.trim()) return;
    setIsSending(true);
    stopListening(true);
    const userMsg = createMessage("user", text);
    const thinkingMsg = createMessage("agent", "Thinking…", "thinking");
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
          .concat(createMessage("agent", `⚠️ ${err.message || "Something went wrong."}`))
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

  useEffect(() => {
    return () => stopListening(true);
  }, []);

  return (
    <div className="page">
      <header className="hero">
        <div className="hero-text">
          <h1>Sales Insight Console</h1>
          <p>Ask questions about your sales data and get concise responses.</p>
        </div>
      </header>

      <main className="layout">
        <section className="chat-panel">
          <div className="chat-header">
            <div>
              <p className="eyebrow">Thread</p>
              <h2>Conversation</h2>
            </div>
            <button type="button" className="ghost" disabled={isSending} onClick={() => setMessages(messages.slice(0, 1))}>
              New chat
            </button>
          </div>
          <div className="messages" ref={listRef}>
            {messages.map((m) => (
              <div key={m.id} className={`bubble ${m.role} ${m.id === "thinking" ? "thinking" : ""}`}>
                <span className="label">{m.role === "user" ? "You" : "Agent"}</span>
                <div dangerouslySetInnerHTML={{ __html: marked.parse(m.text) }} />
              </div>
            ))}
          </div>
        </section>

        <section className="composer">
          <div className="prompt-row">
            <form onSubmit={handleSubmit} className="prompt-form">
              <div className="input-wrap">
                <textarea
                  id="prompt"
                  rows="2"
                  placeholder="Ask about revenue, profit, discounts, regions, or products..."
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
                <div className="actions">
                  <button
                    type="button"
                    className={`mic-btn ${isRecording ? "active" : ""}`}
                    onClick={toggleListening}
                    disabled={isSending}
                    aria-pressed={isRecording}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="icon"
                      aria-hidden="true"
                    >
                      <path d="M12 1a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3Z" />
                      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                      <line x1="12" x2="12" y1="19" y2="23" />
                      <line x1="8" x2="16" y1="23" y2="23" />
                    </svg>
                    <span>{isRecording ? "Stop" : "Speak"}</span>
                  </button>
                  <button type="submit" disabled={isSending}>
                    {isSending ? "Sending..." : "Send"}
                  </button>
                </div>
              </div>
            </form>
            <p className={`hint ${speechError ? "error" : ""}`}>
              {speechError ||
                (isRecording
                  ? "Listening... tap the mic to finish dictation."
                  : "Tip: Press Enter to send, Shift+Enter for a new line.")}
            </p>
          </div>

          <div className="quick-prompts">
            {quickPrompts.map((qp) => (
              <button
                key={qp}
                className="chip"
                type="button"
                onClick={() => setPrompt(qp)}
                disabled={isSending}
              >
                {qp}
              </button>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
