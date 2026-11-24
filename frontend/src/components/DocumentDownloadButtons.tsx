import { useState } from 'react';
import type { ChunkingStrategy } from '../api/datasources';
import { downloadExtractionDocument, type DocumentExportFormat } from '../api/extraction';

const FORMATS: Array<{ format: DocumentExportFormat; label: string }> = [
  { format: 'json', label: 'JSON' },
  { format: 'yaml', label: 'YAML' },
  { format: 'jsonld', label: 'JSON-LD' },
  { format: 'ttl', label: 'TTL' },
  { format: 'shacl', label: 'SHACL report' },
];

export function DocumentDownloadButtons({
  dataPackageId,
  chunkingStrategy,
  chatModel,
  disabled = false,
  onError,
}: {
  dataPackageId: string;
  chunkingStrategy?: ChunkingStrategy;
  chatModel?: string | null;
  disabled?: boolean;
  onError: (message: string) => void;
}) {
  const [pending, setPending] = useState<DocumentExportFormat | null>(null);

  async function download(format: DocumentExportFormat) {
    setPending(format);
    try {
      const { blob, filename } = await downloadExtractionDocument(dataPackageId, format, chunkingStrategy, chatModel);
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
    <div
      className="document-download-buttons"
      title="Downloads the curated document if one was saved, otherwise the generated draft. The SHACL report validates its RDF graph against the registered profile's shapes."
    >
      <span>Download</span>
      {FORMATS.map(({ format, label }) => (
        <button
          key={format}
          className="ghost"
          type="button"
          onClick={() => void download(format)}
          disabled={disabled || pending !== null}
        >
          {pending === format ? `${label}...` : label}
        </button>
      ))}
    </div>
  );
}
