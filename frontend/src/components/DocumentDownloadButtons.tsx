import { useState } from 'react';
import type { ChunkingStrategy } from '../api/datasources';
import { downloadExtractionDocument, type DocumentExportFormat, type DocumentStageRequest } from '../api/extraction';

const FORMATS: Array<{ format: DocumentExportFormat; label: string }> = [
  { format: 'json', label: 'JSON' },
  { format: 'yaml', label: 'YAML' },
  { format: 'jsonld', label: 'JSON-LD' },
  { format: 'ttl', label: 'TTL' },
  { format: 'shacl', label: 'SHACL report' },
];

export function DocumentDownloadButtons({
  dataPackageId,
  stage,
  label,
  title,
  hint,
  chunkingStrategy,
  chatModel,
  disabled = false,
  onError,
}: {
  dataPackageId: string;
  stage: DocumentStageRequest;
  label: string;
  title: string;
  /** Shown instead of the buttons when set, e.g. why nothing can be downloaded yet. */
  hint?: string | null;
  chunkingStrategy?: ChunkingStrategy;
  chatModel?: string | null;
  disabled?: boolean;
  onError: (message: string) => void;
}) {
  const [pending, setPending] = useState<DocumentExportFormat | null>(null);

  async function download(format: DocumentExportFormat) {
    setPending(format);
    try {
      const { blob, filename } = await downloadExtractionDocument(dataPackageId, format, stage, chunkingStrategy, chatModel);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (error) {
      onError(error instanceof Error ? error.message : `Download as ${format} failed.`);
    } finally {
      setPending(null);
    }
  }

  return (
    <div className="document-download-buttons" title={title}>
      <span>{label}</span>
      {hint ? (
        <span className="document-download-hint">{hint}</span>
      ) : (
        FORMATS.map(({ format, label: formatLabel }) => (
          <button
            key={format}
            className="ghost"
            type="button"
            onClick={() => void download(format)}
            disabled={disabled || pending !== null}
          >
            {pending === format ? `${formatLabel}...` : formatLabel}
          </button>
        ))
      )}
    </div>
  );
}
