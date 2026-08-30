// ============================================================
// TruthLens AI - Frontend JavaScript
// ============================================================

// ============================================================
// API CONFIGURATION
// ============================================================
//
// LOCAL:
//   http://127.0.0.1:8000
//
// VERCEL:
//   https://truthlens-ai-backend-mtf8.onrender.com
//
// The frontend automatically selects the correct backend.
// ============================================================

const API_URL =
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1"
        ? "http://127.0.0.1:8000"
        : "https://truthlens-ai-backend-mtf8.onrender.com";

// ==============================================
// DOM ELEMENTS
// ============================================================

const contentInput = document.getElementById("contentInput");
const analyzeButton = document.getElementById("analyzeButton");
const resultContent = document.getElementById("resultContent");


// ============================================================
// SECURITY
// ============================================================

function escapeHtml(value) {
    if (value === null || value === undefined) {
        return "";
    }

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


// ============================================================
// NUMBER / STATUS HELPERS
// ============================================================

function safeNumber(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return 0;
    }

    return Math.max(0, Math.min(100, number));
}


function formatConfidence(value) {
    return `${safeNumber(value).toFixed(2)}%`;
}


function cleanStatus(value) {
    return String(value || "INSUFFICIENT EVIDENCE")
        .replace(/_/g, " ")
        .toUpperCase()
        .trim();
}


// ============================================================
// FINAL VERDICT
// ============================================================

function getFinalVerdict(data) {

    if (
        data &&
        data.final_assessment &&
        data.final_assessment.verdict
    ) {
        return cleanStatus(
            data.final_assessment.verdict
        );
    }

    if (data && data.verdict) {
        return cleanStatus(data.verdict);
    }

    return "INSUFFICIENT EVIDENCE";
}


// ============================================================
// FINAL CONFIDENCE
// ============================================================

function getFinalConfidence(data) {

    if (
        data &&
        data.final_assessment &&
        data.final_assessment.confidence !== undefined
    ) {
        return Number(
            data.final_assessment.confidence
        );
    }

    if (
        data &&
        data.confidence !== undefined
    ) {
        return Number(data.confidence);
    }

    if (
        data &&
        data.evidence_assessment &&
        data.evidence_assessment.score !== undefined
    ) {
        return Number(
            data.evidence_assessment.score
        );
    }

    return 0;
}


// ============================================================
// VERDICT SETTINGS
// ============================================================

function getVerdictSettings(verdict) {

    const normalized = cleanStatus(verdict);

    if (normalized === "SUPPORTED") {

        return {
            icon: "✓",
            color: "text-emerald-400",
            border: "border-emerald-500/30",
            background: "bg-emerald-500/10",
            bar: "bg-emerald-400",
            label: "SUPPORTED",
            description:
                "Relevant external evidence supports the claim."
        };
    }

    if (normalized === "CONTRADICTED") {

        return {
            icon: "✕",
            color: "text-red-400",
            border: "border-red-500/30",
            background: "bg-red-500/10",
            bar: "bg-red-400",
            label: "CONTRADICTED",
            description:
                "Relevant external evidence contradicts the claim."
        };
    }

    return {
        icon: "?",
        color: "text-yellow-400",
        border: "border-yellow-500/30",
        background: "bg-yellow-500/10",
        bar: "bg-yellow-400",
        label: "INSUFFICIENT EVIDENCE",
        description:
            "Available evidence is not sufficiently conclusive."
    };
}


// ============================================================
// PROGRESS BAR
// ============================================================

function renderProgressBar(value, barClass) {

    const percentage = safeNumber(value);

    return `
        <div class="
            w-full
            h-2.5
            rounded-full
            bg-slate-800
            overflow-hidden
        ">
            <div
                class="
                    h-full
                    rounded-full
                    ${barClass}
                    transition-all
                    duration-700
                "
                style="width:${percentage}%"
            ></div>
        </div>
    `;
}


// ============================================================
// LOADING
// ============================================================

function showLoading() {

    resultContent.innerHTML = `
        <div class="text-center py-8">

            <div class="
                text-5xl
                mb-4
                animate-pulse
            ">
                🧠
            </div>

            <p class="
                text-slate-200
                font-semibold
            ">
                TruthLens AI is analyzing...
            </p>

            <p class="
                text-sm
                text-slate-500
                mt-2
            ">
                Retrieving and verifying evidence.
            </p>

        </div>
    `;
}


// ============================================================
// ERROR
// ============================================================

function showError(message) {

    resultContent.innerHTML = `
        <div class="
            p-6
            rounded-2xl
            bg-red-500/10
            border
            border-red-500/20
            text-center
        ">

            <div class="text-4xl mb-3">
                ⚠️
            </div>

            <p class="
                font-semibold
                text-red-400
            ">
                Analysis Failed
            </p>

            <p class="
                text-sm
                text-slate-400
                mt-2
            ">
                ${escapeHtml(message)}
            </p>

        </div>
    `;
}


// ============================================================
// CONFIDENCE VISUALIZATION
// ============================================================

function renderConfidenceVisualization(data, settings) {

    const modelAssessment =
        data.model_assessment || {};

    const evidenceAssessment =
        data.evidence_assessment || {};

    const modelConfidence =
        safeNumber(
            modelAssessment.confidence
        );

    const evidenceScore =
        safeNumber(
            evidenceAssessment.score
        );

    const finalConfidence =
        safeNumber(
            getFinalConfidence(data)
        );

    return `
        <div class="
            mt-6
            p-6
            rounded-2xl
            bg-slate-950
            border
            border-slate-800
            text-left
        ">

            <div class="
                flex
                items-center
                justify-between
                mb-6
            ">

                <div>

                    <p class="
                        text-xs
                        uppercase
                        tracking-wider
                        text-blue-400
                    ">
                        Confidence Overview
                    </p>

                    <h4 class="
                        text-lg
                        font-semibold
                        text-white
                        mt-1
                    ">
                        Evidence-based assessment
                    </h4>

                </div>

                <span class="
                    text-xs
                    text-slate-500
                ">
                    0–100%
                </span>

            </div>


            <!-- FINAL CONFIDENCE -->

            <div class="mb-6">

                <div class="
                    flex
                    items-center
                    justify-between
                    mb-2
                ">

                    <span class="
                        text-sm
                        text-slate-300
                        font-medium
                    ">
                        Final Confidence
                    </span>

                    <span class="
                        text-sm
                        font-bold
                        ${settings.color}
                    ">
                        ${formatConfidence(finalConfidence)}
                    </span>

                </div>

                ${renderProgressBar(
                    finalConfidence,
                    settings.bar
                )}

            </div>


            <!-- MODEL CONFIDENCE -->

            <div class="mb-6">

                <div class="
                    flex
                    items-center
                    justify-between
                    mb-2
                ">

                    <span class="
                        text-sm
                        text-slate-400
                    ">
                        AI Model Confidence
                    </span>

                    <span class="
                        text-sm
                        text-blue-400
                        font-semibold
                    ">
                        ${formatConfidence(modelConfidence)}
                    </span>

                </div>

                ${renderProgressBar(
                    modelConfidence,
                    "bg-blue-400"
                )}

            </div>


            <!-- EVIDENCE STRENGTH -->

            <div>

                <div class="
                    flex
                    items-center
                    justify-between
                    mb-2
                ">

                    <span class="
                        text-sm
                        text-slate-400
                    ">
                        Evidence Strength
                    </span>

                    <span class="
                        text-sm
                        text-purple-400
                        font-semibold
                    ">
                        ${formatConfidence(evidenceScore)}
                    </span>

                </div>

                ${renderProgressBar(
                    evidenceScore,
                    "bg-purple-400"
                )}

            </div>

        </div>
    `;
}


// ============================================================
// VERIFICATION BREAKDOWN
// ============================================================

function renderVerificationBreakdown(data) {

    const evidence =
        data.evidence_assessment || {};

    const verification =
        evidence.verification || {};

    const supported =
        safeNumber(
            verification.supported_score
        );

    const contradicted =
        safeNumber(
            verification.contradicted_score
        );

    const neutral =
        safeNumber(
            verification.neutral_score
        );

    return `
        <div class="
            mt-5
            p-6
            rounded-2xl
            bg-slate-950
            border
            border-slate-800
            text-left
        ">

            <div class="mb-5">

                <p class="
                    text-xs
                    uppercase
                    tracking-wider
                    text-blue-400
                ">
                    Semantic Verification
                </p>

                <h4 class="
                    text-lg
                    font-semibold
                    text-white
                    mt-1
                ">
                    Verification Breakdown
                </h4>

            </div>


            <!-- SUPPORTED -->

            <div class="mb-4">

                <div class="
                    flex
                    justify-between
                    mb-2
                ">

                    <span class="
                        text-sm
                        text-emerald-400
                    ">
                        Supported
                    </span>

                    <span class="
                        text-sm
                        text-slate-400
                    ">
                        ${formatConfidence(supported)}
                    </span>

                </div>

                ${renderProgressBar(
                    supported,
                    "bg-emerald-400"
                )}

            </div>


            <!-- CONTRADICTED -->

            <div class="mb-4">

                <div class="
                    flex
                    justify-between
                    mb-2
                ">

                    <span class="
                        text-sm
                        text-red-400
                    ">
                        Contradicted
                    </span>

                    <span class="
                        text-sm
                        text-slate-400
                    ">
                        ${formatConfidence(contradicted)}
                    </span>

                </div>

                ${renderProgressBar(
                    contradicted,
                    "bg-red-400"
                )}

            </div>


            <!-- NEUTRAL -->

            <div>

                <div class="
                    flex
                    justify-between
                    mb-2
                ">

                    <span class="
                        text-sm
                        text-yellow-400
                    ">
                        Neutral
                    </span>

                    <span class="
                        text-sm
                        text-slate-400
                    ">
                        ${formatConfidence(neutral)}
                    </span>

                </div>

                ${renderProgressBar(
                    neutral,
                    "bg-yellow-400"
                )}

            </div>

        </div>
    `;
}


// ============================================================
// EXPLAINABILITY
// ============================================================

function generateExplanation(data, verdict) {

    const model =
        data.model_assessment || {};

    const evidence =
        data.evidence_assessment || {};

    const verification =
        evidence.verification || {};

    const finalAssessment =
        data.final_assessment || {};

    const sources =
        Array.isArray(evidence.sources)
            ? evidence.sources
            : [];

    const finalConfidence =
        safeNumber(
            getFinalConfidence(data)
        );

    const modelConfidence =
        safeNumber(
            model.confidence
        );

    const evidenceScore =
        safeNumber(
            evidence.score
        );

    const supportedScore =
        safeNumber(
            verification.supported_score
        );

    const contradictedScore =
        safeNumber(
            verification.contradicted_score
        );

    const neutralScore =
        safeNumber(
            verification.neutral_score
        );


    const backendBasis =
        finalAssessment.basis
            ? String(finalAssessment.basis).trim()
            : "";


    const modelVerdict =
        cleanStatus(
            model.verdict || "UNKNOWN"
        );


    const evidenceStatus =
        cleanStatus(
            evidence.status ||
            "EVIDENCE_UNAVAILABLE"
        );


    const sourceCount =
        sources.length;


    const sourceSummary =
        sourceCount > 0
            ? `TruthLens retrieved ${sourceCount} external evidence source${sourceCount === 1 ? "" : "s"} for verification.`
            : "TruthLens did not retrieve usable external evidence.";


    let verificationSummary = "";


    if (
        supportedScore >= contradictedScore &&
        supportedScore >= neutralScore
    ) {

        verificationSummary =
            `Semantic verification favored supporting evidence with a verification score of ${formatConfidence(
                supportedScore
            )}.`;

    } else if (
        contradictedScore >= supportedScore &&
        contradictedScore >= neutralScore
    ) {

        verificationSummary =
            `Semantic verification favored contradictory evidence with a verification score of ${formatConfidence(
                contradictedScore
            )}.`;

    } else {

        verificationSummary =
            `Semantic verification was primarily neutral with a neutral score of ${formatConfidence(
                neutralScore
            )}.`;
    }


    const modelSummary =
        `The AI model initially assessed the claim as ${modelVerdict} with ${formatConfidence(
            modelConfidence
        )} confidence.`;


    const evidenceSummary =
        `External evidence was assessed as ${evidenceStatus} with an evidence strength of ${formatConfidence(
            evidenceScore
        )}.`;


    let reconciliation = "";


    if (verdict === "SUPPORTED") {

        if (
            modelVerdict !== "SUPPORTED" &&
            evidenceScore > modelConfidence
        ) {

            reconciliation =
                `The initial AI prediction was not sufficiently confident to determine the final result. ` +
                `TruthLens therefore relied more strongly on the retrieved external evidence and semantic verification. ` +
                `Because the evidence strongly supports the claim, the evidence-based assessment takes precedence over the low-confidence AI prediction.`;

        } else {

            reconciliation =
                `The AI assessment and external evidence were combined with semantic verification to determine the final result. ` +
                `The available evidence supports the claim.`;
        }

    } else if (verdict === "CONTRADICTED") {

        if (
            modelVerdict !== "CONTRADICTED" &&
            evidenceScore > modelConfidence
        ) {

            reconciliation =
                `The initial AI prediction was not sufficiently confident to determine the final result. ` +
                `TruthLens relied more strongly on external evidence and semantic verification. ` +
                `Because the retrieved evidence contradicts the claim, the evidence-based assessment determines the final result.`;

        } else {

            reconciliation =
                `The AI assessment, external evidence, and semantic verification were combined to determine the final result. ` +
                `The available evidence contradicts the claim.`;
        }

    } else {

        reconciliation =
            `TruthLens could not establish a sufficiently reliable conclusion because the available model assessment, evidence, and semantic verification were not conclusive enough.`;
    }


    let explanation = "";


    if (verdict === "SUPPORTED") {

        explanation =
            `TruthLens reached a SUPPORTED result through a multi-stage verification process. ` +
            `${modelSummary} ` +
            `${evidenceSummary} ` +
            `${verificationSummary} ` +
            `${reconciliation} ` +
            `${sourceSummary} ` +
            `The combined assessment produced a final confidence of ${formatConfidence(
                finalConfidence
            )}.`;

    } else if (verdict === "CONTRADICTED") {

        explanation =
            `TruthLens reached a CONTRADICTED result through a multi-stage verification process. ` +
            `${modelSummary} ` +
            `${evidenceSummary} ` +
            `${verificationSummary} ` +
            `${reconciliation} ` +
            `${sourceSummary} ` +
            `The combined assessment produced a final confidence of ${formatConfidence(
                finalConfidence
            )}.`;

    } else {

        explanation =
            `TruthLens reached an INSUFFICIENT EVIDENCE result because the available verification signals were not sufficiently conclusive. ` +
            `${modelSummary} ` +
            `${evidenceSummary} ` +
            `${verificationSummary} ` +
            `${sourceSummary} ` +
            `The final confidence was ${formatConfidence(
                finalConfidence
            )}.`;
    }


    return {

        explanation,

        backendBasis,

        modelSummary,

        evidenceSummary,

        verificationSummary,

        reconciliation,

        sourceSummary,

        modelVerdict,

        evidenceStatus,

        modelConfidence,

        evidenceScore,

        supportedScore,

        contradictedScore,

        neutralScore,

        finalConfidence
    };
}


// ============================================================
// EXPLAINABILITY CARD
// ============================================================

function renderExplainability(data, verdict) {

    const explanation =
        generateExplanation(
            data,
            verdict
        );


    return `
        <div class="
            mt-5
            p-6
            rounded-2xl
            bg-blue-500/5
            border
            border-blue-500/20
            text-left
        ">

            <div class="
                flex
                items-start
                gap-4
            ">

                <div class="
                    shrink-0
                    w-10
                    h-10
                    rounded-full
                    bg-blue-500/10
                    border
                    border-blue-500/20
                    flex
                    items-center
                    justify-center
                    text-blue-400
                ">
                    💡
                </div>


                <div class="flex-1">

                    <p class="
                        text-xs
                        uppercase
                        tracking-wider
                        text-blue-400
                    ">
                        Explainability
                    </p>

                    <h4 class="
                        text-lg
                        font-semibold
                        text-white
                        mt-1
                    ">
                        Why TruthLens reached this result
                    </h4>

                    <p class="
                        text-sm
                        text-slate-400
                        leading-relaxed
                        mt-4
                    ">
                        ${escapeHtml(
                            explanation.explanation
                        )}
                    </p>

                </div>

            </div>


            <div class="
                mt-6
                p-5
                rounded-xl
                bg-slate-950
                border
                border-blue-500/20
            ">

                <p class="
                    text-xs
                    uppercase
                    tracking-wider
                    text-blue-400
                ">
                    Decision Reconciliation
                </p>

                <h5 class="
                    text-base
                    font-semibold
                    text-white
                    mt-2
                ">
                    How the final assessment was determined
                </h5>

                <p class="
                    text-sm
                    text-slate-400
                    leading-relaxed
                    mt-3
                ">
                    ${escapeHtml(
                        explanation.reconciliation
                    )}
                </p>

            </div>


            <div class="
                mt-6
                grid
                md:grid-cols-2
                gap-3
            ">


                <!-- STEP 1 -->

                <div class="
                    p-4
                    rounded-xl
                    bg-slate-950
                    border
                    border-slate-800
                ">

                    <div class="
                        flex
                        items-center
                        justify-between
                    ">

                        <p class="
                            text-xs
                            text-blue-400
                            uppercase
                            tracking-wider
                        ">
                            Step 1
                        </p>

                        <span class="
                            text-xs
                            px-2
                            py-1
                            rounded-full
                            bg-blue-500/10
                            text-blue-400
                        ">
                            AI
                        </span>

                    </div>


                    <h5 class="
                        text-sm
                        font-semibold
                        text-white
                        mt-2
                    ">
                        AI Model
                    </h5>


                    <p class="
                        text-xs
                        text-slate-500
                        mt-2
                        leading-relaxed
                    ">
                        ${escapeHtml(
                            explanation.modelSummary
                        )}
                    </p>

                </div>


                <!-- STEP 2 -->

                <div class="
                    p-4
                    rounded-xl
                    bg-slate-950
                    border
                    border-slate-800
                ">

                    <div class="
                        flex
                        items-center
                        justify-between
                    ">

                        <p class="
                            text-xs
                            text-blue-400
                            uppercase
                            tracking-wider
                        ">
                            Step 2
                        </p>

                        <span class="
                            text-xs
                            px-2
                            py-1
                            rounded-full
                            bg-purple-500/10
                            text-purple-400
                        ">
                            EVIDENCE
                        </span>

                    </div>


                    <h5 class="
                        text-sm
                        font-semibold
                        text-white
                        mt-2
                    ">
                        External Evidence
                    </h5>


                    <p class="
                        text-xs
                        text-slate-500
                        mt-2
                        leading-relaxed
                    ">
                        ${escapeHtml(
                            explanation.evidenceSummary
                        )}
                    </p>

                </div>


                <!-- STEP 3 -->

                <div class="
                    p-4
                    rounded-xl
                    bg-slate-950
                    border
                    border-slate-800
                ">

                    <div class="
                        flex
                        items-center
                        justify-between
                    ">

                        <p class="
                            text-xs
                            text-blue-400
                            uppercase
                            tracking-wider
                        ">
                            Step 3
                        </p>

                        <span class="
                            text-xs
                            px-2
                            py-1
                            rounded-full
                            bg-emerald-500/10
                            text-emerald-400
                        ">
                            VERIFY
                        </span>

                    </div>


                    <h5 class="
                        text-sm
                        font-semibold
                        text-white
                        mt-2
                    ">
                        Semantic Verification
                    </h5>


                    <p class="
                        text-xs
                        text-slate-500
                        mt-2
                        leading-relaxed
                    ">
                        ${escapeHtml(
                            explanation.verificationSummary
                        )}
                    </p>

                </div>


                <!-- STEP 4 -->

                <div class="
                    p-4
                    rounded-xl
                    bg-slate-950
                    border
                    border-emerald-500/20
                ">

                    <div class="
                        flex
                        items-center
                        justify-between
                    ">

                        <p class="
                            text-xs
                            text-emerald-400
                            uppercase
                            tracking-wider
                        ">
                            Step 4
                        </p>

                        <span class="
                            text-xs
                            px-2
                            py-1
                            rounded-full
                            bg-emerald-500/10
                            text-emerald-400
                        ">
                            FINAL
                        </span>

                    </div>


                    <h5 class="
                        text-sm
                        font-semibold
                        text-white
                        mt-2
                    ">
                        Final Decision
                    </h5>


                    <p class="
                        text-xs
                        text-slate-500
                        mt-2
                        leading-relaxed
                    ">
                        External evidence and semantic verification
                        are used to determine the evidence-based
                        final assessment.
                    </p>


                    <div class="
                        mt-3
                        text-sm
                        font-bold
                        ${getVerdictSettings(verdict).color}
                    ">
                        ${escapeHtml(verdict)}
                        ·
                        ${formatConfidence(
                            explanation.finalConfidence
                        )}
                    </div>

                </div>

            </div>


            ${
                explanation.backendBasis
                    ? `
                        <div class="
                            mt-5
                            p-4
                            rounded-xl
                            bg-slate-950
                            border
                            border-slate-800
                        ">

                            <p class="
                                text-xs
                                uppercase
                                tracking-wider
                                text-slate-500
                            ">
                                Backend Assessment Basis
                            </p>

                            <p class="
                                text-sm
                                text-slate-400
                                leading-relaxed
                                mt-2
                            ">
                                ${escapeHtml(
                                    explanation.backendBasis
                                )}
                            </p>

                        </div>
                    `
                    : ""
            }

        </div>
    `;
}


// ============================================================
// BEST EVIDENCE
// ============================================================

function renderBestEvidence(bestEvidence) {

    if (!bestEvidence) {

        return `
            <div class="
                mt-6
                p-5
                rounded-xl
                bg-slate-950
                border
                border-slate-800
            ">

                <p class="
                    text-xs
                    uppercase
                    tracking-wider
                    text-slate-500
                ">
                    Best Evidence
                </p>

                <p class="
                    text-sm
                    text-yellow-400
                    mt-3
                ">
                    No sufficiently conclusive
                    external evidence was available.
                </p>

            </div>
        `;
    }


    const title =
        bestEvidence.title ||
        "Evidence";

    const text =
        bestEvidence.text ||
        "No evidence text available.";

    const source =
        bestEvidence.source ||
        "#";

    const relevance =
        bestEvidence.relevance !== undefined
            ? Number(bestEvidence.relevance)
            : null;


    return `
        <div class="
            mt-6
            p-5
            rounded-xl
            bg-slate-950
            border
            border-slate-800
            text-left
        ">

            <div class="
                flex
                items-center
                justify-between
                gap-3
            ">

                <p class="
                    text-xs
                    uppercase
                    tracking-wider
                    text-blue-400
                ">
                    Best Evidence
                </p>

                ${
                    relevance !== null &&
                    Number.isFinite(relevance)
                        ? `
                            <span class="
                                text-xs
                                px-2.5
                                py-1
                                rounded-full
                                bg-blue-500/10
                                text-blue-400
                            ">
                                Relevance
                                ${relevance.toFixed(1)}%
                            </span>
                        `
                        : ""
                }

            </div>


            <h4 class="
                text-lg
                font-semibold
                text-white
                mt-3
            ">
                ${escapeHtml(title)}
            </h4>


            <p class="
                text-sm
                text-slate-400
                leading-relaxed
                mt-3
            ">
                ${escapeHtml(text)}
            </p>


            ${
                source !== "#"
                    ? `
                        <a
                            href="${escapeHtml(source)}"
                            target="_blank"
                            rel="noopener noreferrer"
                            class="
                                inline-flex
                                items-center
                                mt-4
                                text-sm
                                text-blue-400
                                hover:text-blue-300
                            "
                        >
                            Open Evidence Source
                            <span class="ml-1">↗</span>
                        </a>
                    `
                    : ""
            }

        </div>
    `;
}


// ============================================================
// ALL SOURCES
// ============================================================

function renderSources(sources) {

    if (
        !Array.isArray(sources) ||
        sources.length === 0
    ) {

        return `
            <div class="
                mt-6
                p-5
                rounded-xl
                bg-slate-950
                border
                border-slate-800
                text-left
            ">

                <h4 class="
                    text-lg
                    font-semibold
                    text-white
                ">
                    Evidence Sources
                </h4>

                <p class="
                    text-sm
                    text-slate-500
                    mt-3
                ">
                    No external sources were available.
                </p>

            </div>
        `;
    }


    const cards =
        sources
            .slice(0, 5)
            .map(
                (source, index) => {

                    const title =
                        source.title ||
                        `Source ${index + 1}`;

                    const text =
                        source.text ||
                        "No evidence text available.";

                    const url =
                        source.source ||
                        "#";

                    const relevance =
                        source.relevance !== undefined
                            ? Number(
                                source.relevance
                            )
                            : null;

                    const verification =
                        source.verification ||
                        {};

                    const status =
                        cleanStatus(
                            verification.status ||
                            "NEUTRAL"
                        );


                    let statusColor =
                        "text-slate-400";


                    if (
                        status === "SUPPORTED"
                    ) {

                        statusColor =
                            "text-emerald-400";

                    } else if (
                        status === "CONTRADICTED"
                    ) {

                        statusColor =
                            "text-red-400";
                    }


                    return `
                        <div class="
                            p-5
                            rounded-xl
                            bg-slate-950
                            border
                            border-slate-800
                        ">

                            <div class="
                                flex
                                items-start
                                justify-between
                                gap-4
                            ">

                                <div>

                                    <p class="
                                        text-xs
                                        text-slate-600
                                    ">
                                        Evidence
                                        ${index + 1}
                                    </p>

                                    <h5 class="
                                        font-semibold
                                        text-white
                                        mt-1
                                    ">
                                        ${escapeHtml(title)}
                                    </h5>

                                </div>


                                ${
                                    relevance !== null &&
                                    Number.isFinite(relevance)
                                        ? `
                                            <span class="
                                                shrink-0
                                                text-xs
                                                px-2
                                                py-1
                                                rounded-full
                                                bg-blue-500/10
                                                text-blue-400
                                            ">
                                                ${relevance.toFixed(1)}%
                                                relevance
                                            </span>
                                        `
                                        : ""
                                }

                            </div>


                            <p class="
                                text-sm
                                text-slate-400
                                leading-relaxed
                                mt-3
                            ">
                                ${escapeHtml(text)}
                            </p>


                            <div class="
                                flex
                                flex-wrap
                                gap-2
                                mt-4
                            ">

                                <span class="
                                    text-xs
                                    ${statusColor}
                                    px-2.5
                                    py-1
                                    rounded-full
                                    bg-slate-900
                                    border
                                    border-slate-800
                                ">
                                    ${escapeHtml(status)}
                                </span>


                                ${
                                    verification.confidence !== undefined
                                        ? `
                                            <span class="
                                                text-xs
                                                text-slate-500
                                                px-2.5
                                                py-1
                                                rounded-full
                                                bg-slate-900
                                                border
                                                border-slate-800
                                            ">
                                                Verification:
                                                ${formatConfidence(
                                                    verification.confidence
                                                )}
                                            </span>
                                        `
                                        : ""
                                }

                            </div>


                            ${
                                url !== "#"
                                    ? `
                                        <a
                                            href="${escapeHtml(url)}"
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            class="
                                                inline-flex
                                                mt-4
                                                text-sm
                                                text-blue-400
                                                hover:text-blue-300
                                            "
                                        >
                                            View Source →
                                        </a>
                                    `
                                    : ""
                            }

                        </div>
                    `;
                }
            )
            .join("");


    return `
        <div class="
            mt-6
            text-left
        ">

            <div class="
                flex
                items-center
                justify-between
                mb-3
            ">

                <h4 class="
                    text-lg
                    font-semibold
                    text-white
                ">
                    Evidence Sources
                </h4>

                <span class="
                    text-xs
                    text-slate-500
                ">
                    ${sources.length}
                    source(s)
                </span>

            </div>


            <div class="space-y-3">
                ${cards}
            </div>

        </div>
    `;
}


// ============================================================
// FINAL RESULT
// ============================================================

function renderResult(data) {

    const verdict =
        getFinalVerdict(data);

    const confidence =
        getFinalConfidence(data);

    const settings =
        getVerdictSettings(verdict);

    const content =
        data.content ||
        contentInput.value.trim();

    const model =
        data.model_assessment ||
        {};

    const evidence =
        data.evidence_assessment ||
        {};

    const sources =
        Array.isArray(evidence.sources)
            ? evidence.sources
            : [];

    const bestEvidence =
        evidence.best_evidence ||
        null;


    resultContent.innerHTML = `

        <!-- FINAL VERDICT -->

        <div class="
            p-7
            rounded-2xl
            ${settings.background}
            border
            ${settings.border}
            text-center
        ">

            <div class="
                mx-auto
                w-16
                h-16
                rounded-full
                flex
                items-center
                justify-center
                bg-slate-950/70
                border
                ${settings.border}
            ">

                <span class="
                    text-4xl
                    font-bold
                    ${settings.color}
                ">
                    ${settings.icon}
                </span>

            </div>


            <p class="
                text-xs
                uppercase
                tracking-widest
                text-slate-500
                mt-5
            ">
                TruthLens AI Final Assessment
            </p>


            <h3 class="
                text-3xl
                md:text-4xl
                font-extrabold
                ${settings.color}
                mt-2
            ">
                ${escapeHtml(settings.label)}
            </h3>


            <p class="
                text-sm
                text-slate-400
                mt-3
            ">
                ${escapeHtml(settings.description)}
            </p>


            <div class="
                mt-4
                text-sm
                ${settings.color}
                font-semibold
            ">
                Final Confidence:
                ${formatConfidence(confidence)}
            </div>

        </div>


        <!-- CONFIDENCE -->

        ${renderConfidenceVisualization(
            data,
            settings
        )}


        <!-- VERIFICATION -->

        ${renderVerificationBreakdown(data)}


        <!-- ANALYZED CONTENT -->

        <div class="
            mt-5
            p-5
            rounded-xl
            bg-slate-950
            border
            border-slate-800
            text-left
        ">

            <p class="
                text-xs
                uppercase
                tracking-wider
                text-slate-500
            ">
                Analyzed Content
            </p>


            <p class="
                text-sm
                text-slate-300
                leading-relaxed
                mt-3
                break-words
            ">
                ${escapeHtml(content)}
            </p>

        </div>


        <!-- MODEL + EVIDENCE -->

        <div class="
            grid
            md:grid-cols-2
            gap-4
            mt-5
        ">


            <!-- MODEL -->

            <div class="
                p-5
                rounded-xl
                bg-slate-950
                border
                border-slate-800
                text-left
            ">

                <p class="
                    text-xs
                    uppercase
                    tracking-wider
                    text-slate-500
                ">
                    AI Model Assessment
                </p>


                <p class="
                    text-xl
                    font-bold
                    text-white
                    mt-3
                ">
                    ${escapeHtml(
                        String(
                            model.verdict ||
                            "UNKNOWN"
                        ).toUpperCase()
                    )}
                </p>


                <p class="
                    text-sm
                    text-blue-400
                    mt-2
                ">
                    ${escapeHtml(
                        model.model ||
                        "DistilBERT"
                    )}
                    ·
                    ${formatConfidence(
                        model.confidence || 0
                    )}
                </p>

            </div>


            <!-- EVIDENCE -->

            <div class="
                p-5
                rounded-xl
                bg-slate-950
                border
                border-slate-800
                text-left
            ">

                <p class="
                    text-xs
                    uppercase
                    tracking-wider
                    text-slate-500
                ">
                    Evidence Assessment
                </p>


                <p class="
                    text-xl
                    font-bold
                    ${settings.color}
                    mt-3
                ">
                    ${escapeHtml(
                        cleanStatus(
                            evidence.status ||
                            "EVIDENCE_UNAVAILABLE"
                        )
                    )}
                </p>


                <p class="
                    text-sm
                    text-blue-400
                    mt-2
                ">
                    Evidence Score:
                    ${formatConfidence(
                        evidence.score || 0
                    )}
                </p>

            </div>

        </div>


        <!-- EXPLAINABILITY -->

        ${renderExplainability(
            data,
            verdict
        )}


        <!-- BEST EVIDENCE -->

        ${renderBestEvidence(
            bestEvidence
        )}


        <!-- SOURCES -->

        ${renderSources(
            sources
        )}


        <p class="
            text-xs
            text-slate-600
            text-center
            mt-6
        ">
            TruthLens AI provides an AI-assisted,
            evidence-based assessment and should
            not be treated as an absolute determination
            of factual truth.
        </p>

    `;
}


// ============================================================
// VERIFICATION HISTORY
// ============================================================

const HISTORY_STORAGE_KEY =
    "truthlens_verification_history";

const MAX_HISTORY_ITEMS = 10;


function getVerificationHistory() {

    try {

        const saved =
            localStorage.getItem(
                HISTORY_STORAGE_KEY
            );

        if (!saved) {
            return [];
        }

        const history =
            JSON.parse(saved);

        return Array.isArray(history)
            ? history
            : [];

    } catch (error) {

        console.error(
            "History read error:",
            error
        );

        return [];
    }
}


function saveVerificationHistory(history) {

    try {

        localStorage.setItem(
            HISTORY_STORAGE_KEY,
            JSON.stringify(history)
        );

    } catch (error) {

        console.error(
            "History save error:",
            error
        );
    }
}


function addToVerificationHistory(data) {

    const history =
        getVerificationHistory();

    const claim =
        data.content ||
        contentInput.value.trim();

    if (!claim) {
        return;
    }

    const item = {

        claim: claim,

        verdict:
            getFinalVerdict(data),

        confidence:
            getFinalConfidence(data),

        timestamp:
            new Date().toISOString()
    };


    const filtered =
        history.filter(
            oldItem =>
                oldItem.claim !== claim
        );


    filtered.unshift(item);


    saveVerificationHistory(
        filtered.slice(
            0,
            MAX_HISTORY_ITEMS
        )
    );


    renderVerificationHistory();
}


function getHistoryStyle(verdict) {

    const status =
        cleanStatus(verdict);

    if (status === "SUPPORTED") {

        return {
            icon: "✓",
            color: "text-emerald-400",
            border: "border-emerald-500/20"
        };
    }

    if (status === "CONTRADICTED") {

        return {
            icon: "✕",
            color: "text-red-400",
            border: "border-red-500/20"
        };
    }

    return {
        icon: "?",
        color: "text-yellow-400",
        border: "border-yellow-500/20"
    };
}


function renderVerificationHistory() {

    const history =
        getVerificationHistory();

    let container =
        document.getElementById(
            "verificationHistory"
        );


    if (!container) {

        container =
            document.createElement("div");

        container.id =
            "verificationHistory";

        container.className =
            "max-w-7xl mx-auto px-6 mt-8 mb-8";

        const main =
            document.querySelector("main");

        if (main) {
            main.appendChild(container);
        }
    }


    if (history.length === 0) {

        container.innerHTML = "";

        return;
    }


    const cards =
        history
            .map(item => {

                const style =
                    getHistoryStyle(
                        item.verdict
                    );

                const date =
                    item.timestamp
                        ? new Date(
                            item.timestamp
                        ).toLocaleString()
                        : "";


                return `
                    <div class="
                        p-4
                        rounded-xl
                        bg-slate-950
                        border
                        ${style.border}
                    ">

                        <div class="
                            flex
                            justify-between
                            gap-4
                        ">

                            <div>

                                <p class="
                                    text-sm
                                    text-slate-300
                                ">
                                    ${escapeHtml(
                                        item.claim
                                    )}
                                </p>

                                <p class="
                                    text-xs
                                    text-slate-600
                                    mt-2
                                ">
                                    ${escapeHtml(date)}
                                </p>

                            </div>


                            <div class="
                                text-right
                                shrink-0
                            ">

                                <span class="
                                    text-lg
                                    ${style.color}
                                ">
                                    ${style.icon}
                                </span>

                                <p class="
                                    text-xs
                                    font-semibold
                                    ${style.color}
                                ">
                                    ${escapeHtml(
                                        item.verdict
                                    )}
                                </p>

                                <p class="
                                    text-xs
                                    text-slate-500
                                    mt-1
                                ">
                                    ${formatConfidence(
                                        item.confidence
                                    )}
                                </p>

                            </div>

                        </div>

                    </div>
                `;
            })
            .join("");


    container.innerHTML = `
        <section class="
            bg-slate-900
            border
            border-slate-800
            rounded-2xl
            p-6
            md:p-8
        ">

            <div class="
                flex
                items-center
                justify-between
                mb-5
            ">

                <div>

                    <p class="
                        text-xs
                        uppercase
                        tracking-wider
                        text-blue-400
                    ">
                        TruthLens AI
                    </p>

                    <h3 class="
                        text-xl
                        font-bold
                        text-white
                        mt-1
                    ">
                        Recent Verifications
                    </h3>

                </div>


                <button
                    id="clearHistoryButton"
                    type="button"
                    class="
                        text-xs
                        text-slate-500
                        hover:text-red-400
                    "
                >
                    Clear History
                </button>

            </div>


            <div class="
                grid
                md:grid-cols-2
                gap-3
            ">
                ${cards}
            </div>

        </section>
    `;


    const clearButton =
        document.getElementById(
            "clearHistoryButton"
        );


    if (clearButton) {

        clearButton.addEventListener(
            "click",
            function () {

                localStorage.removeItem(
                    HISTORY_STORAGE_KEY
                );

                renderVerificationHistory();
            }
        );
    }
}


// ============================================================
// ANALYZE CLAIM
// ============================================================

async function analyzeClaim() {

    const content =
        contentInput.value.trim();


    if (!content) {

        resultContent.innerHTML = `
            <div class="text-center py-6">

                <div class="text-4xl mb-4">
                    ⚠️
                </div>

                <p class="text-slate-300">
                    Please enter some content first.
                </p>

            </div>
        `;

        contentInput.focus();

        return;
    }


    analyzeButton.disabled = true;

    analyzeButton.textContent =
        "Analyzing...";


    showLoading();


    try {

        const response =
            await fetch(
                `${API_URL}/analyze`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        content: content
                    })
                }
            );


        if (!response.ok) {

            throw new Error(
                `Backend returned HTTP ${response.status}`
            );
        }


        const data =
            await response.json();


        if (!data.success) {

            throw new Error(
                data.message ||
                "The backend could not analyze the content."
            );
        }


        renderResult(data);

        addToVerificationHistory(data);

    } catch (error) {

        console.error(
            "TruthLens API Error:",
            error
        );

        showError(
            error.message ||
            "Unable to connect to the TruthLens backend."
        );

    } finally {

        analyzeButton.disabled = false;

        analyzeButton.textContent =
            "Analyze Content";
    }
}


// ============================================================
// EVENTS
// ============================================================

if (analyzeButton) {

    analyzeButton.addEventListener(
        "click",
        analyzeClaim
    );
}


if (contentInput) {

    contentInput.addEventListener(
        "keydown",
        function(event) {

            if (
                event.key === "Enter" &&
                (
                    event.ctrlKey ||
                    event.metaKey
                )
            ) {

                event.preventDefault();

                analyzeClaim();
            }

        }
    );
}


document.addEventListener(
    "DOMContentLoaded",
    function() {

        renderVerificationHistory();

    }
);