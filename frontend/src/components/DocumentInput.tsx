interface Props {
  documentText: string;
  question: string;
  onDocumentChange: (text: string) => void;
  onQuestionChange: (text: string) => void;
}

export function DocumentInput({ documentText, question, onDocumentChange, onQuestionChange }: Props) {
  return (
    <div className="space-y-4">
      {/* Document input */}
      <div className="relative">
        <label className="block font-mono text-[10px] font-500 tracking-[0.15em] text-text-muted uppercase mb-2">
          Source Document
        </label>
        <textarea
          value={documentText}
          onChange={(e) => onDocumentChange(e.target.value)}
          placeholder="Paste your reference document here..."
          rows={8}
          className="scope-input w-full px-4 py-3 text-sm leading-relaxed resize-y min-h-[180px]"
        />
        {documentText && (
          <span className="absolute bottom-3 right-3 font-mono text-[10px] text-text-muted">
            {documentText.length.toLocaleString()} chars
          </span>
        )}
      </div>

      {/* Question input */}
      <div>
        <label className="block font-mono text-[10px] font-500 tracking-[0.15em] text-text-muted uppercase mb-2">
          Question / Prompt
        </label>
        <input
          value={question}
          onChange={(e) => onQuestionChange(e.target.value)}
          placeholder="What question was asked about this document?"
          className="scope-input w-full px-4 py-3 text-sm"
        />
      </div>
    </div>
  );
}
