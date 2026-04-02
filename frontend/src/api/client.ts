import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

export interface ClaimResponse {
  claim: string;
  hallucination_score: number;
  tier: 'supported' | 'uncertain' | 'hallucinated';
  verifier_details: {
    nli: Record<string, any>;
    consistency: Record<string, any>;
    similarity: Record<string, any>;
  };
}

export interface AnalyzeResponse {
  id: number | null;
  question: string;
  answer: string;
  scored_claims: ClaimResponse[];
  retrieved_chunks: string[];
  overall_score: number;
}

export interface AnalysisListItem {
  id: number;
  question: string;
  answer: string;
  overall_score: number;
  created_at: string | null;
}

export async function analyzeDocument(documentText: string, question: string): Promise<AnalyzeResponse> {
  const { data } = await api.post<AnalyzeResponse>('/analyze', {
    document_text: documentText,
    question,
  });
  return data;
}

export async function listAnalyses(limit = 50): Promise<AnalysisListItem[]> {
  const { data } = await api.get<AnalysisListItem[]>('/analyses', { params: { limit } });
  return data;
}

export async function submitFeedback(
  analysisId: number,
  claimIndex: number,
  isCorrect: boolean,
  note?: string,
): Promise<void> {
  await api.post('/feedback', {
    analysis_id: analysisId,
    claim_index: claimIndex,
    is_correct: isCorrect,
    note,
  });
}
