const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000";

async function request(
  endpoint,
  options = {}
) {
  const response = await fetch(
    `${API_BASE_URL}${endpoint}`,
    {
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
      ...options,
    }
  );

  let data = null;

  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    const message =
      data?.detail ||
      `Request failed with status ${response.status}.`;

    throw new Error(message);
  }

  return data;
}

export async function healthCheck() {
  return request("/health");
}

export async function generateText({
  prompt,
  maxNewTokens = 50,
  temperature = 1.0,
  topK = null,
}) {
  return request("/generate", {
    method: "POST",
    body: JSON.stringify({
      prompt,
      max_new_tokens: maxNewTokens,
      temperature,
      top_k: topK,
    }),
  });
}

export async function queryRAG(
  question
) {
  return request("/rag", {
    method: "POST",
    body: JSON.stringify({
      question,
    }),
  });
}

export async function runAgent(
  query
) {
  return request("/agent", {
    method: "POST",
    body: JSON.stringify({
      query,
    }),
  });
}

export async function remember({
  content,
  memoryType = "fact",
  metadata = {},
}) {
  return request("/memory", {
    method: "POST",
    body: JSON.stringify({
      content,
      memory_type: memoryType,
      metadata,
    }),
  });
}

export async function recallMemory({
  query,
  topK = 5,
}) {
  return request("/memory/recall", {
    method: "POST",
    body: JSON.stringify({
      query,
      top_k: topK,
    }),
  });
}

export async function clearMemory() {
  return request("/memory", {
    method: "DELETE",
  });
}