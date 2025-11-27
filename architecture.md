graph LR
    %% ==========================================
    %% 1. Style Definitions (Modern Blues/Greens)
    %% ==========================================
    classDef client fill:#f5f7f9,stroke:#607d8b,stroke-width:2px,color:#37474f,rx:5,ry:5;
    classDef api fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px,color:#0d47a1,rx:5,ry:5;
    classDef agent fill:#e8f5e9,stroke:#43a047,stroke-width:2px,color:#1b5e20,rx:5,ry:5;
    classDef tool fill:#e0f2f1,stroke:#00897b,stroke-width:2px,color:#004d40,rx:5,ry:5;
    classDef data fill:#eceff1,stroke:#546e7a,stroke-width:2px,color:#263238,shape:cylinder;
    classDef vis fill:#fff8e1,stroke:#ffa000,stroke-width:2px,color:#ff6f00,rx:5,ry:5;

    %% ==========================================
    %% 2. Node Definitions
    %% ==========================================
    
    User(Client<br/>HTTP Caller):::client
    
    subgraph Backend [Backend System]
        direction LR
        
        API[API Layer<br/>FastAPI / Azure Func<br/><i>CORS Enabled</i>]:::api
        
        Orch[Agent Orchestrator<br/>LangGraph + LangChain<br/><i>ChatOpenAI</i>]:::agent
        
        subgraph Analysis [Data & Analytics]
            direction TB
            Pandas[Analytics Tool<br/>python_df<br/><i>In-memory DataFrame</i>]:::tool
            DB[(Data Layer<br/>SQLite<br/><i>data/retail_sales.sqlite</i>)]:::data
        end
        
        subgraph Output [Artifacts]
            direction TB
            MatPlot[Visualization<br/>Matplotlib]:::vis
            Storage[Artifacts Folder<br/><i>/artifacts (Timestamped)</i><br/>Served via Static/CDN]:::vis
        end
    end

    %% ==========================================
    %% 3. Data Flow & Connections
    %% ==========================================

    %% Entry
    User -->|POST /query| API
    API -->|Pass Request| Orch

    %% Logic Loop
    Orch <-->|Tool Loop| Pandas
    
    %% Data Interaction
    Pandas <-->|Query / Seed Synthetic if Missing| DB
    
    %% Visualization Trigger
    Pandas -.->|Trigger on 'Chart' keywords| MatPlot
    MatPlot -->|Generate PNG| Storage
    
    %% Return Path
    Storage -.->|Return Public URL| API
    Orch -->|Final Answer + Context| API
    API -->|JSON Response| User

    %% ==========================================
    %% 4. Annotations (Context)
    %% ==========================================
    
    %% Using a non-connected node for config notes to keep diagram clean
    Note[<b>Configuration</b><br/>Swap DB/Table via Env Vars<br/>or CLI Flags]:::client
    style Note stroke-dasharray: 5 5
    
    %% Link note visually to the relevant area (Data Layer)
    Note -.- DB