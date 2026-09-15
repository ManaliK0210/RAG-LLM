function ChatMessage({
  role,
  content,
}) {
  return (
    <div
      className={`chat-message ${role}`}
    >
      <div className="message-role">
        {role === "user"
          ? "You"
          : "RAGForge"}
      </div>

      <div className="message-content">
        {content}
      </div>
    </div>
  );
}

export default ChatMessage;