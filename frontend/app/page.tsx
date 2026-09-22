"use client";

import { useState, useEffect, useRef } from "react";
import { sendChatMessage, fetchHealth, generatePlan, PlanTask } from "./services/api";
import { Zap } from "lucide-react";

interface Message {
  sender: "user" | "coach";
  text: string;
  citations?: string[];
  isPlanEligible?: boolean;   // true if message looks like a plan/schedule
  planAdded?: boolean;        // true once user clicked to generate checklist
  tokens_used?: number;
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
}

interface TrackerDot {
  id: string;
  sessionId: string;           // scoped to chat session
  title: string;
  icon: string;
  tasks: (PlanTask & { done: boolean })[];
}

// ─── Keyword detection: does this coach response look like a plan? ───
function isPlanMessage(text: string): boolean {
  const keywords = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "week", "daily", "routine", "schedule", "plan", "day 1", "day 2",
    "training plan", "hydration plan", "recovery plan", "step 1", "step 2",
    "first", "then", "next", "finally", "phase",
  ];
  const lower = text.toLowerCase();
  return keywords.some((kw) => lower.includes(kw));
}

export default function Home() {
  // ─── Runner Profile States ───
  const [experienceLevel, setExperienceLevel] = useState<string>("beginner");
  const [distanceTier, setDistanceTier] = useState<string>("marathon");
  const [age, setAge] = useState<string>("28");
  const [gender, setGender] = useState<string>("female");

  // ─── Multi-Chat Session States ───
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [apiHealthy, setApiHealthy] = useState<boolean>(false);

  // ─── Tracker Dots (floating checklist buttons) ───
  const [trackers, setTrackers] = useState<TrackerDot[]>([]);
  const [openTrackerId, setOpenTrackerId] = useState<string | null>(null);
  const [generatingPlanForMsgIdx, setGeneratingPlanForMsgIdx] = useState<number | null>(null);
  const [showTokensForMsg, setShowTokensForMsg] = useState<{ [key: number]: boolean }>({});

  // ─── UI Panel states ───
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(true);
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [activeCitation, setActiveCitation] = useState<string | null>(null);
  const [inputMessage, setInputMessage] = useState<string>("");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // ─── Initial Load & Persistence ───
  useEffect(() => {
    // Check backend health, and start polling if it doesn't respond instantly (sleeping on Render free-tier)
    const checkHealth = () => {
      fetchHealth()
        .then(() => setApiHealthy(true))
        .catch(() => {
          setApiHealthy(false);
          // Retry every 4 seconds until the server wakes up
          const interval = setInterval(() => {
            fetchHealth()
              .then(() => {
                setApiHealthy(true);
                clearInterval(interval);
              })
              .catch(() => {
                setApiHealthy(false);
              });
          }, 4000);
          return () => clearInterval(interval);
        });
    };

    checkHealth();

    const savedTheme = localStorage.getItem("coach_theme") as "dark" | "light";
    if (savedTheme) setTheme(savedTheme);

    const savedProfile = localStorage.getItem("coach_profile");
    if (savedProfile) {
      try {
        const parsed = JSON.parse(savedProfile);
        if (parsed.age) setAge(parsed.age);
        if (parsed.gender) setGender(parsed.gender);
        if (parsed.experienceLevel) setExperienceLevel(parsed.experienceLevel);
        if (parsed.distanceTier) setDistanceTier(parsed.distanceTier);
      } catch (e) { }
    }

    // Load trackers
    const savedTrackers = localStorage.getItem("coach_trackers");
    if (savedTrackers) {
      try { setTrackers(JSON.parse(savedTrackers)); } catch (e) { }
    }

    const savedSessions = localStorage.getItem("coach_sessions");
    if (savedSessions) {
      try {
        const parsed = JSON.parse(savedSessions) as ChatSession[];
        if (parsed.length > 0) {
          setSessions(parsed);
          setActiveSessionId(parsed[0].id);
          return;
        }
      } catch (e) { }
    }

    const initialSessionId = `chat_${Date.now()}`;
    const defaultSessions: ChatSession[] = [
      {
        id: initialSessionId,
        title: "Initial Consultation",
        messages: [
          {
            sender: "coach",
            text: "Hello! I am your AI Running Coach. Tell me about your training goals, or ask me any question about hydration, nutrition, stretching, or injury prevention!",
          },
        ],
      },
    ];
    setSessions(defaultSessions);
    setActiveSessionId(initialSessionId);
    localStorage.setItem("coach_sessions", JSON.stringify(defaultSessions));
  }, []);

  useEffect(() => {
    const profile = { age, gender, experienceLevel, distanceTier };
    localStorage.setItem("coach_profile", JSON.stringify(profile));
  }, [age, gender, experienceLevel, distanceTier]);

  useEffect(() => {
    if (sessions.length > 0) localStorage.setItem("coach_sessions", JSON.stringify(sessions));
  }, [sessions]);

  useEffect(() => {
    if (trackers.length > 0) localStorage.setItem("coach_trackers", JSON.stringify(trackers));
    else localStorage.removeItem("coach_trackers");
  }, [trackers]);

  useEffect(() => { localStorage.setItem("coach_theme", theme); }, [theme]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [sessions, activeSessionId, isLoading]);

  const currentSession = sessions.find((s) => s.id === activeSessionId);
  const activeMessages = currentSession ? currentSession.messages : [];

  // Active session tracker dots only
  const sessionTrackers = trackers.filter((t) => t.sessionId === activeSessionId);

  // ─── New Chat ───
  const handleNewChat = () => {
    const newId = `chat_${Date.now()}`;
    const newSession: ChatSession = {
      id: newId,
      title: "New Run Session",
      messages: [
        {
          sender: "coach",
          text: `Hi there! Ready to plan your training? Ask me anything about running as a ${experienceLevel} level athlete.`,
        },
      ],
    };
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newId);
    setOpenTrackerId(null);
  };

  // ─── Delete Chat ───
  const handleDeleteChat = (idToDelete: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (sessions.length <= 1) {
      alert("You must keep at least one chat session.");
      return;
    }
    const updated = sessions.filter((s) => s.id !== idToDelete);
    setSessions(updated);
    if (activeSessionId === idToDelete) setActiveSessionId(updated[0].id);
  };

  // ─── Send Message ───
  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    const userQuery = inputMessage;
    setInputMessage("");

    setSessions((prev) =>
      prev.map((s) => {
        if (s.id !== activeSessionId) return s;
        const title =
          s.title === "New Run Session" || s.title === "Initial Consultation"
            ? userQuery.substring(0, 24) + (userQuery.length > 24 ? "..." : "")
            : s.title;
        return { ...s, title, messages: [...s.messages, { sender: "user", text: userQuery }] };
      })
    );
    setIsLoading(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await sendChatMessage(userQuery, experienceLevel, distanceTier, age, gender, controller.signal);
      const coachText = response.answer;

      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== activeSessionId) return s;
          return {
            ...s,
            messages: [
              ...s.messages,
              {
                sender: "coach",
                text: coachText,
                citations: response.citations,
                isPlanEligible: isPlanMessage(coachText),
                planAdded: false,
                tokens_used: response.tokens_used,
              },
            ],
          };
        })
      );
    } catch (err: any) {
      const errorMessage =
        err.name === "AbortError"
          ? "⚠️ Response generation was stopped by the user."
          : `❌ Error: ${err.message}. Make sure your FastAPI backend and Qdrant docker are running.`;

      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== activeSessionId) return s;
          return { ...s, messages: [...s.messages, { sender: "coach", text: errorMessage }] };
        })
      );
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  const handleStopGeneration = () => {
    if (abortControllerRef.current) abortControllerRef.current.abort();
  };

  // ─── Generate Plan Checklist ───
  const handleGeneratePlan = async (msgIndex: number, coachText: string) => {
    setGeneratingPlanForMsgIdx(msgIndex);
    try {
      const plan = await generatePlan(coachText, experienceLevel, distanceTier);

      const newTracker: TrackerDot = {
        id: `tracker_${Date.now()}`,
        sessionId: activeSessionId,
        title: plan.title,
        icon: plan.icon,
        tasks: plan.tasks.map((t) => ({ ...t, done: false })),
      };
      setTrackers((prev) => [...prev, newTracker]);

      // Mark message as planAdded
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== activeSessionId) return s;
          const msgs = s.messages.map((m, i) => (i === msgIndex ? { ...m, planAdded: true } : m));
          return { ...s, messages: msgs };
        })
      );
    } catch (err) {
      // Silently fail — don't break the UI
    } finally {
      setGeneratingPlanForMsgIdx(null);
    }
  };

  // ─── Tracker Task Toggle ───
  const toggleTrackerTask = (trackerId: string, taskIndex: number) => {
    setTrackers((prev) =>
      prev.map((t) => {
        if (t.id !== trackerId) return t;
        const tasks = t.tasks.map((task, i) => (i === taskIndex ? { ...task, done: !task.done } : task));
        return { ...t, tasks };
      })
    );
  };

  // ─── Delete Tracker ───
  // Also resets planAdded on the corresponding message so the ☑ icon reappears
  const deleteTracker = (trackerId: string) => {
    const removed = trackers.find((t) => t.id === trackerId);
    setTrackers((prev) => prev.filter((t) => t.id !== trackerId));
    if (openTrackerId === trackerId) setOpenTrackerId(null);

    // Re-enable the checklist icon on the session's messages so user can regenerate
    if (removed) {
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== removed.sessionId) return s;
          return {
            ...s,
            messages: s.messages.map((m) =>
              m.sender === "coach" && m.planAdded ? { ...m, planAdded: false } : m
            ),
          };
        })
      );
    }
  };

  // ─── Color Palette Helper (Royal Blue, Green, Orange) ───
  const colors = {
    bg: theme === "dark" ? "bg-[#07090e] text-[#f3f4f6]" : "bg-[#faf7ee] text-[#1e293b]",
    sidebar: theme === "dark" ? "bg-[#0d111a]/45 backdrop-blur-2xl border-white/5" : "bg-[#f4eed8]/50 backdrop-blur-2xl border-black/5",
    card: theme === "dark" ? "bg-[#151b26]/35 border-[#1e293b]/40 text-[#d1d5db]" : "bg-[#eae3cb]/25 border-[#dfd7be]/40 text-[#334155]",
    input: theme === "dark" ? "bg-[#07090e]/70 border-[#273549] text-white focus:border-blue-500" : "bg-white/80 border-[#dfd7be] text-slate-900 focus:border-blue-500",
    header: theme === "dark" ? "bg-[#0d111a]/80 border-[#1e293b]/60" : "bg-[#faf7ee]/80 border-[#dfd7be]",
    bubbleUser: "bg-blue-600 text-white rounded-br-none shadow-md shadow-blue-900/10",
    bubbleCoach: theme === "dark" ? "bg-[#151b26]/70 border-[#1e293b]/60 text-[#d1d5db] rounded-bl-none" : "bg-white border-[#dfd7be] text-slate-700 rounded-bl-none shadow-sm",
    citationBtn: theme === "dark" ? "bg-orange-950/40 text-orange-400 border-orange-900/50 hover:bg-orange-900/70" : "bg-orange-50 text-orange-600 border border-orange-200 hover:bg-orange-100",
    activeChat: theme === "dark" ? "bg-[#1d4ed8]/20 text-blue-300 border-blue-800/40" : "bg-[#1d4ed8]/15 text-blue-800 border-blue-300/60",
    inactiveChat: theme === "dark" ? "hover:bg-slate-800/40 text-gray-400 hover:text-white" : "hover:bg-slate-200/50 text-slate-500 hover:text-slate-950",
    trackerPopup: theme === "dark" ? "bg-[#0d111a] border-[#1e293b] text-[#f3f4f6]" : "bg-white border-[#dfd7be] text-[#1e293b]",
  };

  return (
    <div className={`flex h-screen w-screen overflow-hidden font-sans transition-colors duration-300 ${colors.bg}`}>

      {/* ─── LEFT SIDEBAR ─── */}
      <aside
        className={`h-full flex flex-col justify-between shrink-0 transition-all duration-300 overflow-y-auto border-r ${colors.sidebar} ${isSidebarOpen ? "w-72 p-5 translate-x-0 opacity-100" : "w-0 p-0 opacity-0 -translate-x-full"
          }`}
      >
        {isSidebarOpen && (
          <div className="flex flex-col gap-5">
            <div className="flex items-center justify-between pb-3 border-b border-dashed border-gray-700/30">
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></div>
                <h1 className="text-lg font-extrabold tracking-tight bg-gradient-to-r from-blue-400 via-green-400 to-orange-400 bg-clip-text text-transparent">
                  RUNNABLE !
                </h1>
              </div>
              <button
                onClick={() => setIsSidebarOpen(false)}
                className="p-1 hover:bg-slate-800/60 rounded cursor-pointer transition text-gray-400 hover:text-white"
                title="Collapse Panel"
              >
                ◀
              </button>
            </div>

            <div className={`p-4 rounded-xl border flex flex-col gap-3.5 shadow-xl shadow-black/5 ${colors.card}`}>
              <h2 className="text-[10px] font-extrabold text-emerald-400 uppercase tracking-widest">Runner Profile</h2>

              <div className="flex gap-3">
                <div className="flex-1 flex flex-col gap-0.5">
                  <label className="text-[10px] font-semibold text-gray-400">Age</label>
                  <input type="number" value={age} onChange={(e) => setAge(e.target.value)}
                    className={`w-full rounded-md px-2 py-1 text-xs focus:outline-none border ${colors.input}`} />
                </div>
                <div className="flex-1 flex flex-col gap-0.5">
                  <label className="text-[10px] font-semibold text-gray-400">Gender</label>
                  <select value={gender} onChange={(e) => setGender(e.target.value)}
                    className={`w-full rounded-md px-2 py-1 text-xs focus:outline-none border ${colors.input}`}>
                    <option value="female">Female</option>
                    <option value="male">Male</option>
                    <option value="other">Other</option>
                  </select>
                </div>
              </div>

              <div className="flex flex-col gap-0.5">
                <label className="text-[10px] font-semibold text-gray-400">Experience</label>
                <select value={experienceLevel} onChange={(e) => setExperienceLevel(e.target.value)}
                  className={`w-full rounded-md px-2 py-1 text-xs focus:outline-none border font-semibold ${colors.input}`}>
                  <option value="beginner">Beginner</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="advanced">Advanced</option>
                </select>
              </div>

              <div className="flex flex-col gap-0.5">
                <label className="text-[10px] font-semibold text-gray-400">Goal Distance</label>
                <select value={distanceTier} onChange={(e) => setDistanceTier(e.target.value)}
                  className={`w-full rounded-md px-2 py-1 text-xs focus:outline-none border font-semibold ${colors.input}`}>
                  <option value="5k">5K run</option>
                  <option value="10k">10K run</option>
                  <option value="half_marathon">Half Marathon (21km)</option>
                  <option value="marathon">Marathon (42km)</option>
                </select>
              </div>
            </div>

            <button onClick={handleNewChat}
              className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-2 rounded-xl text-xs transition cursor-pointer shadow-md shadow-blue-800/10 flex items-center justify-center gap-1.5">
              ➕ New Chat Consultation
            </button>

            <div className="flex flex-col gap-1.5 mt-2">
              <h3 className="text-[10px] font-extrabold text-blue-400 uppercase tracking-widest">Consultations</h3>
              <div className="flex flex-col gap-1 max-h-48 overflow-y-auto pr-1">
                {sessions.map((s) => (
                  <div key={s.id} onClick={() => { setActiveSessionId(s.id); setOpenTrackerId(null); }}
                    className={`flex items-center justify-between text-xs px-3 py-2.5 rounded-lg border cursor-pointer transition ${s.id === activeSessionId ? colors.activeChat : colors.inactiveChat}`}>
                    <span className="truncate pr-2 font-medium">{s.title}</span>
                    <button onClick={(e) => handleDeleteChat(s.id, e)}
                      className="text-gray-500 hover:text-red-400 text-xs px-1" title="Delete Session">✕</button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {isSidebarOpen && (
          <div className="mt-4 pt-3 border-t border-gray-700/20 flex items-center gap-1.5 text-[10px] text-gray-400">
            <span className={`w-2 h-2 rounded-full ${apiHealthy ? "bg-emerald-500" : "bg-red-500"}`}></span>
            <span>{apiHealthy ? "Server Online" : "Server Offline"}</span>
          </div>
        )}
      </aside>

      {/* ─── RIGHT CHAT PANEL ─── */}
      <main className="flex-1 flex flex-col justify-between h-full relative">

        {/* Header */}
        <header className={`px-5 py-3.5 border-b flex items-center justify-between transition-colors duration-300 ${colors.header}`}>
          <div className="flex items-center gap-3">
            {!isSidebarOpen && (
              <button onClick={() => setIsSidebarOpen(true)}
                className="p-2 border rounded-lg cursor-pointer hover:bg-blue-600 hover:text-white hover:border-blue-600 transition border-gray-500 text-gray-400 flex items-center justify-center"
                title="Expand Profile Settings">
                <svg className="w-4 h-4 fill-none stroke-current" strokeWidth="2" viewBox="0 0 24 24">
                  <rect width="18" height="18" x="3" y="3" rx="2"></rect>
                  <path d="M9 3v18"></path>
                </svg>
              </button>
            )}
            <span className="text-xs font-semibold text-gray-400 tracking-wider uppercase">Your Running Buddy</span>
          </div>
          <button onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="w-8 h-8 rounded-full border flex items-center justify-center text-sm cursor-pointer transition border-gray-600 hover:border-blue-500 hover:text-blue-500"
            title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}>
            {theme === "dark" ? "☀️" : "🌙"}
          </button>
        </header>

        {/* Waking up banner */}
        {!apiHealthy && (
          <div className="bg-amber-500/10 border-b border-amber-500/20 px-5 py-3 text-xs text-amber-500 flex items-center justify-between animate-pulse transition-all">
            <div className="flex items-center gap-2">
              <span className="flex h-2 w-2 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
              </span>
              <span>
                <strong>Note:</strong> Connecting to cloud server. If it is sleeping, waking up takes ~45 seconds. Please wait...
              </span>
            </div>
          </div>
        )}

        {/* Chat Message Feed */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {activeMessages.map((msg, index) => (
            <div key={index}
              className={`flex flex-col max-w-2xl ${msg.sender === "user" ? "ml-auto items-end" : "mr-auto items-start"}`}>

              {/* Coach bubble wrapper with hover to reveal checklist icon */}
              {msg.sender === "coach" ? (
                <div className="relative group">
                  <div className={`p-4 rounded-2xl text-sm leading-relaxed border ${colors.bubbleCoach}`}>
                    {msg.text.split("\n").map((para, i) => (
                      <p key={i} className={para.trim() ? "mb-2" : "h-2"} style={{ whiteSpace: "pre-wrap" }}>{para}</p>
                    ))}

                    {msg.citations && msg.citations.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-700/20 flex flex-wrap items-center gap-2">
                        <span className="text-[10px] text-orange-400 uppercase tracking-widest font-extrabold">Sources:</span>
                        {msg.citations.map((cite, cIdx) => (
                          <button key={cIdx} onClick={() => setActiveCitation(cite)}
                            className={`border px-2 py-0.5 rounded text-[10px] font-bold tracking-wide transition cursor-pointer ${colors.citationBtn}`}>
                            📄 {cite}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* ─── Minimal checklist icon (hover reveals tooltip) ─── */}
                  {msg.isPlanEligible && !msg.planAdded && (
                    <button
                      onClick={() => handleGeneratePlan(index, msg.text)}
                      title="Generate checklist to track your progress"
                      disabled={generatingPlanForMsgIdx === index}
                      className="absolute -top-3 -right-3 w-7 h-7 rounded-full bg-emerald-600 hover:bg-emerald-500 text-white flex items-center justify-center cursor-pointer transition shadow-lg shadow-emerald-900/30 opacity-0 group-hover:opacity-100 text-xs disabled:animate-spin"
                    >
                      {generatingPlanForMsgIdx === index ? "⟳" : "☑"}
                    </button>
                  )}

                  {/* "✓ added to tracker" confirmation */}
                  {msg.planAdded && (
                    <span className="text-[10px] text-emerald-500 font-semibold mt-1 ml-1">✓ added to tracker</span>
                  )}
                  
                  {/* Token Count Icon */}
                  {msg.tokens_used && (
                    <div className="mt-2 flex justify-end items-center relative">
                      {showTokensForMsg[index] && (
                        <span className="text-[10px] text-gray-500 dark:text-gray-400 mr-2 bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded-full shadow-sm animate-in fade-in zoom-in duration-200">
                          ⚡ {msg.tokens_used} tokens
                        </span>
                      )}
                      <button 
                        onClick={() => setShowTokensForMsg(prev => ({ ...prev, [index]: !prev[index] }))}
                        className="text-yellow-500 hover:text-yellow-600 transition-colors drop-shadow-sm hover:scale-110 duration-200"
                        title="View Token Usage"
                      >
                        <Zap size={16} fill="currentColor" />
                      </button>
                    </div>
                  )}
                </div>
              ) : (
                /* User bubble */
                <div className={`p-4 rounded-2xl text-sm leading-relaxed border ${colors.bubbleUser}`}>
                  {msg.text.split("\n").map((para, i) => (
                    <p key={i} className={para.trim() ? "mb-2" : "h-2"} style={{ whiteSpace: "pre-wrap" }}>{para}</p>
                  ))}
                </div>
              )}
            </div>
          ))}

          {/* Typing indicator with pulsing red stop dot */}
          {isLoading && (
            <div className="mr-auto items-start flex flex-col max-w-lg">
              <div className={`border p-4 rounded-2xl rounded-bl-none flex items-center gap-3 ${colors.bubbleCoach}`}>
                <span className="text-sm opacity-80">Working</span>
                <span className="flex gap-1">
                  <span className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce delay-100"></span>
                  <span className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce delay-200"></span>
                  <span className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce delay-300"></span>
                </span>
                <div className="relative flex items-center justify-center w-6 h-6">
                  <button onClick={handleStopGeneration} title="Stop response generation"
                    className="w-2.5 h-2.5 bg-red-600 rounded-full cursor-pointer hover:scale-125 transition relative z-10">
                    <span className="absolute inline-flex h-full w-full rounded-full bg-red-500 opacity-75 animate-ping"></span>
                  </button>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <form
          onSubmit={handleSend}
          className={`p-6 border-t ${colors.header}`}
        >
          <div className="flex gap-4 max-w-4xl mx-auto items-end">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={(e) => {
                // Shift+Enter → insert newline; plain Enter → submit
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  if (!isLoading && inputMessage.trim()) handleSend(e as any);
                }
              }}
              placeholder={`Ask Coach about running as a ${experienceLevel}...`}
              rows={1}
              className={`flex-1 rounded-xl px-4 py-3 text-sm focus:outline-none border resize-none overflow-hidden ${colors.input}`}
              style={{ minHeight: "48px", maxHeight: "140px", height: "auto", overflowY: "auto" }}
              onInput={(e) => {
                const el = e.currentTarget;
                el.style.height = "auto";
                el.style.height = Math.min(el.scrollHeight, 140) + "px";
              }}
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !inputMessage.trim()}
              className="bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-50 px-6 py-3 rounded-xl text-sm font-semibold transition cursor-pointer"
            >
              Send
            </button>
          </div>
        </form>

        {/* ─── FLOATING TRACKER DOTS (bottom-right, scoped to active session) ─── */}
        <div className="absolute bottom-36 right-6 flex flex-col-reverse gap-3 z-40">
          {sessionTrackers.map((tracker) => {
            const done = tracker.tasks.filter((t) => t.done).length;
            const total = tracker.tasks.length;
            const isOpen = openTrackerId === tracker.id;

            return (
              <div key={tracker.id} className="relative flex flex-col items-end">
                {/* Expanded Tracker Popup Card */}
                {isOpen && (
                  <div className={`absolute bottom-12 right-0 w-72 rounded-2xl border shadow-2xl p-4 mb-1 ${colors.trackerPopup}`}>
                    <div className="flex items-center justify-between mb-2">
                      <div>
                        <h4 className="font-bold text-sm">{tracker.title}</h4>
                        <p className="text-[10px] text-gray-400">{done} of {total} done</p>
                      </div>
                      <button onClick={() => setOpenTrackerId(null)}
                        className="text-gray-400 hover:text-white text-sm cursor-pointer">✕</button>
                    </div>
                    <div className="flex flex-col gap-2 mt-3">
                      {tracker.tasks.map((task, tIdx) => (
                        <label key={tIdx} className="flex items-center gap-2.5 cursor-pointer group/task">
                          <input type="checkbox" checked={task.done}
                            onChange={() => toggleTrackerTask(tracker.id, tIdx)}
                            className="accent-blue-500 w-3.5 h-3.5 rounded cursor-pointer" />
                          <span className={`text-xs leading-tight transition ${task.done ? "line-through opacity-40" : ""}`}>
                            {task.label}
                          </span>
                        </label>
                      ))}
                    </div>
                  </div>
                )}

                {/* The Floating Dot Button */}
                <div className="relative group/dot">
                  <button
                    onClick={() => setOpenTrackerId(isOpen ? null : tracker.id)}
                    className="w-11 h-11 rounded-full bg-white dark:bg-[#151b26] border border-gray-200 dark:border-[#1e293b] shadow-lg flex items-center justify-center text-lg hover:scale-105 transition cursor-pointer"
                  >
                    {tracker.icon}
                  </button>

                  {/* Progress badge */}
                  <span className="absolute -bottom-1 -right-1 bg-blue-600 text-white text-[9px] font-bold rounded-full px-1.5 py-0.5 leading-tight">
                    {done}/{total}
                  </span>

                  {/* ✕ Delete button on hover */}
                  <button
                    onClick={() => deleteTracker(tracker.id)}
                    title="Remove this tracker"
                    className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full bg-red-500 text-white text-[9px] font-bold flex items-center justify-center cursor-pointer opacity-0 group-hover/dot:opacity-100 transition"
                  >
                    ✕
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {/* ─── CITATION DRAWER ─── */}
        {activeCitation && (
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm z-50 flex justify-end">
            <div className={`w-full max-w-md h-full border-l p-6 flex flex-col justify-between ${colors.sidebar}`}>
              <div>
                <div className="flex justify-between items-center border-b border-gray-700/20 pb-4 mb-4">
                  <h3 className="font-bold text-orange-400">📄 Source Details</h3>
                  <button onClick={() => setActiveCitation(null)} className="text-gray-400 hover:text-white text-lg font-bold cursor-pointer">✕</button>
                </div>
                <div className={`p-4 rounded-xl border ${colors.card}`}>
                  <p className="text-xs text-gray-500 font-semibold uppercase tracking-wider mb-2">Source File</p>
                  <p className="text-sm font-bold text-blue-400 mb-4">{activeCitation}</p>
                  <p className="text-xs text-gray-500 font-semibold uppercase tracking-wider mb-2">Verified Content</p>
                  <p className="text-sm leading-relaxed p-3 rounded-lg bg-black/20 border border-gray-700/20">
                    This file contains the verified sports training, nutrition, and medical guidelines used by the coach to ground this response.
                  </p>
                </div>
              </div>
              <button onClick={() => setActiveCitation(null)}
                className={`w-full py-3 rounded-xl text-sm font-semibold transition cursor-pointer ${theme === "dark" ? "bg-[#1e293b] hover:bg-[#273549] text-white" : "bg-gray-200 hover:bg-gray-300 text-slate-800"}`}>
                Close Details
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
