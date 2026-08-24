/**
 * Demo corpus and golden evaluation set.
 *
 * The eval set is only meaningful because `relevant_doc_ids` are ground truth
 * assigned by hand — Precision@K and MRR measure retrieval against *these*
 * labels. Generating the labels from the retriever's own output would make the
 * metrics circular and always excellent.
 *
 * Chunk ids are `{doc_id}_{contentHash}`, so the labels below name doc_ids and
 * the harness matches on the prefix. Content-addressed chunk ids change when
 * the text changes; doc ids do not.
 */

export interface Document {
    doc_id: string;
    title: string;
    text: string;
}

export const CORPUS: Document[] = [
    {
        doc_id: 'chunking',
        title: 'Chunking Strategy',
        text:
            'Chunking splits a document into passages small enough that each embedding represents a single idea. ' +
            'A whole document embedded as one vector averages all of its topics together and matches nothing well. ' +
            'Overlapping windows prevent a sentence spanning a boundary from being cut in half, which would leave ' +
            'neither passage able to answer a question about it. Chunk size trades retrieval precision against how ' +
            'much surrounding context each passage carries. A common starting point is 500 characters with 100 ' +
            'characters of overlap, tuned against an evaluation set rather than chosen by intuition.'
    },
    {
        doc_id: 'bm25',
        title: 'BM25 Keyword Ranking',
        text:
            'BM25 ranks documents by term frequency and inverse document frequency. The k1 parameter controls term ' +
            'frequency saturation, so a word appearing twenty times is not judged twenty times more relevant than one ' +
            'appearing once. The b parameter controls document length normalisation, preventing long documents from ' +
            'winning simply by containing more words. BM25 excels at rare exact terms and identifiers, where a high ' +
            'inverse document frequency makes a match unmissable.'
    },
    {
        doc_id: 'embeddings',
        title: 'Dense Embeddings',
        text:
            'An embedding maps text into a vector space where distance reflects meaning rather than wording. Two ' +
            'passages sharing no vocabulary sit close together if they express the same idea, which is why semantic ' +
            'search handles paraphrase where keyword search cannot. Embeddings blur rare tokens: an unusual proper ' +
            'noun contributes little to the vector, which is exactly the case keyword search handles best.'
    },
    {
        doc_id: 'cosine',
        title: 'Cosine Similarity',
        text:
            'Cosine similarity measures the angle between two vectors and ignores their magnitude, which makes it ' +
            'scale invariant. A long document and a short one on the same subject therefore score alike. The formula ' +
            'is the dot product divided by the product of both Euclidean norms. Euclidean distance behaves ' +
            'differently because magnitude contributes to the result, and dot product similarity is magnitude ' +
            'sensitive by construction.'
    },
    {
        doc_id: 'hybrid',
        title: 'Hybrid Retrieval',
        text:
            'Hybrid retrieval runs keyword and semantic search together and merges the results. The two fail in ' +
            'opposite directions: keyword search misses paraphrase, semantic search blurs rare identifiers. Weighted ' +
            'fusion normalises both score sets to a common range before blending them, typically weighting the vector ' +
            'arm more heavily. Fusion should union the two candidate sets rather than intersect them, because a ' +
            'result found by only one retriever is precisely what the other was going to miss.'
    },
    {
        doc_id: 'rrf',
        title: 'Reciprocal Rank Fusion',
        text:
            'Reciprocal Rank Fusion merges ranked lists using only rank position, ignoring the underlying scores. ' +
            'Each list contributes one divided by k plus the rank. Because it never compares raw scores, RRF is ' +
            'robust when the systems being merged produce values on incomparable scales. The k parameter damps how ' +
            'sharply early ranks dominate: a small k rewards placing first in one list, a large k rewards appearing ' +
            'in both lists at all. Sixty is the conventional default.'
    },
    {
        doc_id: 'evaluation',
        title: 'RAG Evaluation Metrics',
        text:
            'Precision at K measures what fraction of the top K retrieved documents are relevant, dividing by K so ' +
            'that returning fewer results than requested is not rewarded. Mean reciprocal rank returns one divided ' +
            'by the position of the first relevant document, capturing how quickly a user reaches a useful result. ' +
            'Recall at K measures how many of all known relevant documents appear in the top K. These retrieval ' +
            'metrics are exact arithmetic over ground-truth labels.'
    },
    {
        doc_id: 'llm-judge',
        title: 'LLM as Judge',
        text:
            'An LLM judge scores generated answers without human labels. Faithfulness asks whether every claim in the ' +
            'answer is supported by the retrieved context; an answer that correctly reports the context is ' +
            'insufficient is fully faithful. Relevance asks whether the answer addresses the question asked, ' +
            'independent of whether it is factually correct. Judge scores are one model grading another and are ' +
            'best used to detect regressions between runs rather than as absolute measures of quality.'
    },
    {
        doc_id: 'caching',
        title: 'Semantic Caching',
        text:
            'A semantic cache keys responses by the meaning of a query rather than its exact text, so that ' +
            'differently worded versions of the same question share one cached answer. Incoming queries are embedded ' +
            'and compared against cached query vectors by cosine similarity; a match above the threshold returns the ' +
            'stored response. The threshold is a safety parameter: too low and distinct questions collide, returning ' +
            'a confidently wrong cached answer. Entries carry a time to live so stale answers expire.'
    },
    {
        doc_id: 'rate-limiting',
        title: 'Rate Limiting',
        text:
            'A sliding window rate limiter stores the timestamp of each request and counts those falling inside the ' +
            'window. This avoids the boundary problem of fixed windows, where a client can spend two full windows of ' +
            'budget across a single boundary. Limits should be enforced before any expensive work begins, so a ' +
            'rejected request costs nothing. A rejected request must also not consume budget, or a blocked client ' +
            'extends its own lockout by retrying.'
    },
    {
        doc_id: 'injection',
        title: 'Prompt Injection Defense',
        text:
            'In retrieval augmented generation the injection usually arrives through the corpus rather than the user ' +
            'query, because retrieved documents reach the model wearing the same clothes as legitimate context. ' +
            'Defenses layer: sanitise user input, fence untrusted content with delimiters, neutralise ' +
            'instruction-shaped phrases inside retrieved text, restate the operative rule after the context so it is ' +
            'the last thing read, and validate the output before returning it. No layer is complete on its own, and ' +
            'the strongest containment is limiting what the system can do at all.'
    },
    {
        doc_id: 'self-rag',
        title: 'Self-RAG and Corrective RAG',
        text:
            'Self-RAG lets the model decide whether retrieval is needed at all, rather than retrieving ' +
            'unconditionally, and grades retrieved passages for relevance before using them. Corrective RAG adds an ' +
            'evaluator that classifies the retrieved set as correct, incorrect, or ambiguous, and falls back to a ' +
            'broader source such as web search when the local index is judged insufficient. Both patterns exist ' +
            'because retrieving irrelevant context can degrade an answer that the model would otherwise get right.'
    }
];

/**
 * Golden evaluation set. Twelve cases spanning the query types that stress
 * different parts of the pipeline: exact identifiers, paraphrase, multi-hop,
 * and one question the corpus deliberately cannot answer.
 */
export interface GoldenCase {
    query: string;
    relevant_doc_ids: string[];
    kind: 'exact-term' | 'conceptual' | 'multi-doc' | 'unanswerable';
    note?: string;
}

export const GOLDEN_SET: GoldenCase[] = [
    { query: 'What does the k1 parameter control?', relevant_doc_ids: ['bm25'], kind: 'exact-term' },
    { query: 'What is the conventional default value for k in RRF?', relevant_doc_ids: ['rrf'], kind: 'exact-term' },
    { query: 'How is cosine similarity calculated?', relevant_doc_ids: ['cosine'], kind: 'exact-term' },
    { query: 'Why split documents into overlapping pieces?', relevant_doc_ids: ['chunking'], kind: 'conceptual' },
    { query: 'Why does document length not change the score?', relevant_doc_ids: ['cosine'], kind: 'conceptual' },
    { query: 'How can a search engine understand a question phrased differently from the document?', relevant_doc_ids: ['embeddings'], kind: 'conceptual' },
    { query: 'What stops someone hiding commands inside a document the system reads?', relevant_doc_ids: ['injection'], kind: 'conceptual' },
    { query: 'How do you avoid paying for the same question twice?', relevant_doc_ids: ['caching'], kind: 'conceptual' },
    { query: 'Why merge keyword and semantic search instead of picking one?', relevant_doc_ids: ['hybrid', 'embeddings', 'bm25'], kind: 'multi-doc' },
    { query: 'How do you measure whether retrieval and generation are working?', relevant_doc_ids: ['evaluation', 'llm-judge'], kind: 'multi-doc' },
    { query: 'When should a system skip retrieval entirely?', relevant_doc_ids: ['self-rag'], kind: 'conceptual' },
    {
        query: 'What is the current stock price of Alphabet?',
        relevant_doc_ids: [],
        kind: 'unanswerable',
        note: 'The corpus cannot answer this. A faithful system says so; a hallucinating one invents a number.'
    }
];
