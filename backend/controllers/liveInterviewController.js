const { generateLocalQuestionBank, enrichQuestionBank } = require('../services/bankGenerator');
const Interview = require('../models/Interview');
const User = require('../models/User');
const llm = require('../services/interviewLLMService');

// ============================================================
// HELPERS
// ============================================================

function asArray(value) {
  if (Array.isArray(value)) {
    return value
      .map(String)
      .map(x => x.trim())
      .filter(Boolean);
  }

  if (
    value === undefined ||
    value === null ||
    value === ''
  ) {
    return [];
  }

  return String(value)
    .split(/[,;+]/)
    .map(x => x.trim())
    .filter(Boolean);
}

// ============================================================
// WAIT / RETRY FOR PYTHON AI AGENT
// ============================================================
//
// Python interview.py can take several seconds to start.
// Node may receive the frontend request before port 5100
// is ready.
//
// This helper retries only connection-refused errors.
// Other errors are immediately passed through.
//

async function callWithRetry(
  fn,
  options = {}
) {
  const maxAttempts =
    Number(options.maxAttempts || 30);

  const delayMs =
    Number(options.delayMs || 1000);

  for (
    let attempt = 1;
    attempt <= maxAttempts;
    attempt++
  ) {
    try {
      return await fn();
    } catch (error) {
      const code =
        error?.code ||
        error?.cause?.code;

      const isConnectionRefused =
        code === 'ECONNREFUSED' ||
        String(error?.message || '')
          .includes('ECONNREFUSED');

      // If this is not a startup/network
      // connection problem, don't hide it.
      if (!isConnectionRefused) {
        throw error;
      }

      if (
        attempt >= maxAttempts
      ) {
        throw error;
      }

      console.log(
        `AI Interview Agent not ready. ` +
        `Retrying ${attempt}/${maxAttempts} ` +
        `in ${delayMs}ms...`
      );

      await new Promise(
        resolve =>
          setTimeout(
            resolve,
            delayMs
          )
      );
    }
  }

  throw new Error(
    'AI Interview Agent could not be reached.'
  );
}

// ============================================================
// SETUP NORMALIZATION
// ============================================================

function cleanSetup(
  body = {},
  user = null
) {
  const types = asArray(
    body.types ??
    body.interview_types ??
    body.interviewTypes
  );

  const normalizedTypes =
    types.length
      ? types
      : asArray(
          body.interviewType
        );

  const profile =
    body.profile ||
    user?.profile ||
    null;

  const useResume = Boolean(
    body.useResume ??
    body.use_resume
  );

  const setup = {
    // --------------------------------------------------------
    // CORE INTERVIEW SETUP
    // --------------------------------------------------------

    domain: String(
      body.domain ??
      body.primary_domain ??
      ''
    ).trim(),

    subDomain: String(
      body.subDomain ??
      body.sub_domain ??
      body.sub ??
      ''
    ).trim(),

    role: String(
      body.role ??
      body.specific_role ??
      ''
    ).trim(),

    objective: String(
      body.objective ??
      body.interview_objective ??
      ''
    ).trim(),

    // --------------------------------------------------------
    // INTERVIEW TYPES
    // --------------------------------------------------------

    types: normalizedTypes,

    interview_types:
      normalizedTypes,

    interviewType: String(
      body.interviewType ??
      normalizedTypes.join(', ') ??
      ''
    ).trim(),

    // --------------------------------------------------------
    // INTERVIEW DETAILS
    // --------------------------------------------------------

    stage: String(
      body.stage ??
      ''
    ).trim(),

    companyType: String(
      body.companyType ??
      body.company_type ??
      ''
    ).trim(),

    company: String(
      body.company ??
      body.company_name ??
      ''
    ).trim(),

    duration: Number(
      body.duration ??
      body.duration_min ??
      45
    ),

    difficulty: String(
      body.difficulty ??
      ''
    ).trim(),

    language: String(
      body.language ??
      'English'
    ).trim(),

    // --------------------------------------------------------
    // RESUME
    // --------------------------------------------------------

    useResume,

    use_resume: useResume,

    // --------------------------------------------------------
    // CUSTOM TOPICS
    // --------------------------------------------------------

    customTopics:
      Array.isArray(
        body.customTopics ??
        body.custom_topics
      )
        ? (
            body.customTopics ??
            body.custom_topics
          )
        : String(
            body.customTopics ??
            body.custom_topics ??
            ''
          ).trim(),

    // --------------------------------------------------------
    // CODING / SQL
    // --------------------------------------------------------

    codingLanguage: String(
      body.codingLanguage ??
      body.coding_language ??
      'python'
    ).trim(),

    requiresCoding: Boolean(
      body.requiresCoding ??
      body.requires_coding
    ),

    requires_coding: Boolean(
      body.requiresCoding ??
      body.requires_coding
    ),

    requiresSql: Boolean(
      body.requiresSql ??
      body.requires_sql
    ),

    requires_sql: Boolean(
      body.requiresSql ??
      body.requires_sql
    ),

    isTechnical: Boolean(
      body.isTechnical ??
      body.is_technical
    ),

    is_technical: Boolean(
      body.isTechnical ??
      body.is_technical
    ),

    techStack:
      body.techStack ??
      body.tech_stack ??
      [],

    // --------------------------------------------------------
    // PROFILE
    // --------------------------------------------------------

    profile,

    // --------------------------------------------------------
    // IMPORTANT:
    // Resume context is ONLY included when the user selected
    // "Use Resume".
    // --------------------------------------------------------

    resume: useResume
      ? (
          body.resume ||
          profile?.resume ||
          null
        )
      : null,
  };

  return setup;
}

// ============================================================
// CLASSIFY INTERVIEW
// ============================================================

function classifyInterview(
  setup
) {
  const types =
    setup.types.map(
      x => x.toLowerCase()
    );

  const primaryDomain = `${setup.domain || ''} ${setup.subDomain || ''} ${setup.role || ''}`;
  const softwareDomain = /software|technology|technical|developer|engineer|programming|computer science|backend|frontend|full.?stack|data engineer|devops/i.test(primaryDomain);
  const technical =
    setup.isTechnical || softwareDomain ||
    types.some(t => /technical|coding\s*\/\s*dsa|sql\s*\/\s*database|system design/i.test(t));

  const wantsCoding = setup.requiresCoding || softwareDomain ||
    types.some(t => /technical|coding|dsa/i.test(t));

  const wantsSql = setup.requiresSql || softwareDomain ||
    /sql|database/i.test(primaryDomain) ||
    types.some(t => /technical|sql|database/i.test(t));

  return {
    technical,
    wantsCoding,
    wantsSql,
  };
}

// ============================================================
// BUILD INTERVIEW PLAN
// ============================================================

function buildPlan(
  setup
) {
  const total =
    Math.max(
      6,
      Math.round(
        Number(
          setup.duration || 45
        ) / 3
      )
    );

  const {
    technical,
    wantsCoding,
    wantsSql,
  } =
    classifyInterview(
      setup
    );

  // ----------------------------------------------------------
  // NORMAL / NON-TECHNICAL INTERVIEW
  // ----------------------------------------------------------

  if (!technical) {
    return {
      total,
      verbal: total,
      coding: 0,
      sql: 0,
    };
  }

  // ----------------------------------------------------------
  // TECHNICAL INTERVIEW
  // Roughly 60% verbal + 40% technical
  // ----------------------------------------------------------

  const technicalUnits =
    Math.max(
      2,
      Math.round(
        total * 0.4
      )
    );

  const verbal =
    total -
    technicalUnits;

  let sql =
    wantsSql
      ? Math.max(
          1,
          Math.round(
            technicalUnits *
              0.4
          )
        )
      : 0;

  let coding =
    wantsCoding
      ? technicalUnits - sql
      : 0;

  if (
    !wantsCoding &&
    wantsSql
  ) {
    sql =
      technicalUnits;

    coding = 0;
  }

  if (
    wantsCoding &&
    !wantsSql
  ) {
    coding =
      technicalUnits;

    sql = 0;
  }

  return {
    total,
    verbal,
    coding,
    sql,
  };
}

// ============================================================
// CONVERT MONGODB HISTORY TO LLM HISTORY
// ============================================================

function historyFrom(
  interview
) {
  return interview.answers.map(
    a => ({
      section: a.section,
      question: a.question,
      questionPayload:
        a.questionPayload,
      answer: a.answer,
      transcript: a.transcript,
      evaluation:
        a.evaluation,
      score: a.score,
    })
  );
}

// ============================================================
// AVERAGE SCORE
// ============================================================

function averageScore(
  answers
) {
  if (!Array.isArray(answers) || !answers.length) return 0;
  const scored =
    answers
      .map(
        a => Number(a.score ?? a.evaluation?.score ?? 0)
      )
      .filter(
        Number.isFinite
      );

  return scored.length
    ? Number(
        (
          scored.reduce(
            (a, b) =>
              a + b,
            0
          ) /
          scored.length
        ).toFixed(2)
      )
    : 0;
}

// ============================================================
// START INTERVIEW
// ============================================================

exports.start =
  async (req, res) => {
    try {
      // ------------------------------------------------------
      // LOAD USER
      // ------------------------------------------------------

      const user =
        await User.findById(
          req.userId
        ).lean();

      if (!user) {
        return res
          .status(404)
          .json({
            message:
              'User not found.',
          });
      }

      if (
        !user.profileCompleted
      ) {
        return res
          .status(403)
          .json({
            message:
              'Complete your profile first.',
          });
      }

      // ------------------------------------------------------
      // NORMALIZE SETUP
      // ------------------------------------------------------

      const setup =
        cleanSetup(
          req.body,
          user
        );

      if (
        !setup.domain ||
        !setup.subDomain ||
        !setup.role ||
        !setup.objective ||
        !setup.types.length
      ) {
        return res
          .status(400)
          .json({
            message:
              'Domain, subdomain, role, objective and interview type are required.',
          });
      }

      // ------------------------------------------------------
      // BUILD PLAN
      // ------------------------------------------------------

      const plan =
        buildPlan(setup);

      // ------------------------------------------------------
      // CREATE INTERVIEW
      // ------------------------------------------------------

      const interview =
        await Interview.create({
          userId:
            req.userId,

          setup,

          status:
            'active',

          startedAt:
            new Date(),

          questionCount:
            plan.total,
        });

      // Preload interview context, not a fixed list of exact questions.
      // Only the opening introduction is deterministic; later questions are
      // generated dynamically from the candidate's actual answers/evaluations.
      const questionBank = [{
        id: 1, difficultyTier: 'easy', type: 'verbal', section: 'verbal',
        questionType: 'introduction', category: 'Introduction / Background',
        text: 'Please introduce yourself and walk me through your background, relevant experience, and one project you personally contributed to.',
        intent: 'Evaluate communication, background, ownership and clarity.'
      }];
      const q = questionBank[0];
      interview.questionBank = questionBank;
      await interview.save();

      // Warm the resume/domain RAG context in the background while the
      // candidate answers the introduction. Never delay interview startup.
      void llm.preloadInterviewContext({ setup }).then((r) => {
        console.log(`Interview RAG preload ${r?.ok ? 'ready' : 'fallback'} for ${interview._id}`);
      }).catch((e) => console.warn('Background interview RAG preload failed:', e?.message || e));

      return res.json({
        interviewId: String(interview._id),
        setup,
        plan,
        question: q,
        questionBank,
      });
    } catch (e) {
      console.error(
        'Start interview failed:',
        e
      );

      return res
        .status(500)
        .json({
          message:
            e?.message ||
            'Failed to start interview',
        });
    }
  };

// ============================================================
// ANSWER INTERVIEW QUESTION
// ============================================================

exports.answer =
  async (req, res) => {
    try {
      // ------------------------------------------------------
      // LOAD INTERVIEW
      // ------------------------------------------------------

      const interview =
        await Interview.findOne({
          _id:
            req.params.id,

          userId:
            req.userId,
        });

      if (!interview) {
        return res
          .status(404)
          .json({
            message:
              'Interview not found.',
          });
      }

      if (
        interview.status !==
        'active'
      ) {
        return res
          .status(400)
          .json({
            message:
              'Interview is not active.',
          });
      }

      // ------------------------------------------------------
      // PLAN + HISTORY
      // ------------------------------------------------------

      const plan =
        buildPlan(
          interview.setup
        );

      const history =
        historyFrom(
          interview
        );

      // ------------------------------------------------------
      // CURRENT ANSWER
      // ------------------------------------------------------

      const question =
        String(
          req.body.question ||
          ''
        ).trim();

      const answer =
        String(
          req.body.answer ??
          req.body.transcript ??
          ''
        ).trim();

      const transcript =
        String(
          req.body.transcript ??
          answer
        ).trim();

      const section =
        [
          'verbal',
          'coding',
          'sql',
        ].includes(
          req.body.section
        )
          ? req.body.section
          : 'verbal';

      const codingSubmission =
        req.body
          .codingSubmission ||
        null;

      if (!question) {
        return res
          .status(400)
          .json({
            message:
              'Question is required.',
          });
      }

      const skipped = Boolean(req.body.skipped) || /^(skip|pass|i don['’]?t know|i do not know|idk|no idea)(\b.*)?$/i.test(answer);

      if (
        !answer &&
        !codingSubmission && !skipped
      ) {
        return res
          .status(400)
          .json({
            message:
              'Answer is required.',
          });
      }

      // ------------------------------------------------------
      // EVALUATION PAYLOAD
      // ------------------------------------------------------

      const payload = {
        setup:
          interview.setup,

        plan,

        history,

        questionIndex:
          interview.answers
            .length + 1,

        currentQuestion:
          question,

        answer,

        transcript,

        section,

        codingSubmission,
      };

      // ------------------------------------------------------
      // EVALUATE ANSWER
      //
      // Retry because the Python agent can temporarily
      // restart or take time to become available.
      // ------------------------------------------------------

      const result = skipped
        ? { evaluation: { score: 0, correctness: 'Skipped / not answered', clarity: 'No answer provided', strengths: [], missingPoints: ['Candidate did not provide an answer.'], followUp: false, followUpReason: 'Candidate skipped or indicated they did not know the answer.', followUpQuestion: '', summary: 'Skipped question recorded as 0/10.' }, score: 0, followUp: false }
        : await callWithRetry(() => llm.evaluateAnswer(payload), { maxAttempts: 30, delayMs: 1000 });

      const evaluation =
        result?.evaluation ||
        result ||
        {};

      const score =
        Number(
          result?.score ??
          evaluation?.score ??
          0
        );

      const followUp =
        Boolean(
          result?.followUp ??
          evaluation?.followUp
        );

      const followUpReason =
        String(
          result?.followUpReason ??
          evaluation?.followUpReason ??
          ''
        );

      // ------------------------------------------------------
      // SAVE ANSWER
      // ------------------------------------------------------

      const isCrossQuestion = Boolean(
        req.body.crossQuestion ||
        req.body.questionPayload?.crossQuestion ||
        req.body.questionType === 'cross_question' ||
        req.body.questionPayload?.questionType === 'cross_question' ||
        /cross-question|follow-up/i.test(req.body.questionPayload?.category || '') ||
        /cross-question|follow-up/i.test(req.body.category || '')
      );

      const savedAnswer = {
        questionNumber: interview.answers.length + 1,
        section,
        questionType: isCrossQuestion ? 'cross_question' : (result?.questionType || section),
        crossQuestion: isCrossQuestion,
        question,
        questionPayload: req.body.questionPayload || null,
        answer: skipped ? (answer || '[Skipped — candidate did not know / chose to pass]') : answer,
        skipped,
        transcript,
        codingSubmission,
        followUp,
        followUpReason,
        evaluation,
        score,
        answeredAt: new Date(),
      };

      interview.answers.push(
        savedAnswer
      );

      // ------------------------------------------------------
      // SHOULD INTERVIEW END?
      // ------------------------------------------------------

      const completedMainQuestions = interview.answers.filter(a => a.questionType !== 'cross_question' && !a.crossQuestion).length;
      const reachedPlan = completedMainQuestions >= plan.total;

      const shouldEnd =
        reachedPlan ||
        Boolean(
          result?.endInterview ??
          evaluation?.endInterview
        );

      // ------------------------------------------------------
      // COMPLETE INTERVIEW
      // ------------------------------------------------------

      if (shouldEnd) {
        interview.status =
          'completed';

        interview.completedAt =
          new Date();

        interview.durationSeconds =
          interview.startedAt
            ? Math.round(
                (
                  Date.now() -
                  interview.startedAt.getTime()
                ) / 1000
              )
            : 0;

        interview.finalScore =
          averageScore(
            interview.answers
          );

        interview.summary = {
          averageScore:
            interview.finalScore,

          questionCount:
            interview.answers.length,

          completedBecause:
            reachedPlan
              ? 'planned_question_count'
              : 'ai_end_interview',

          lastSummary:
            result?.summary ??
            evaluation?.summary ??
            '',
        };

        await interview.save();

        return res.json({
          done: true,

          evaluation,

          score,

          summary:
            interview.summary,

          interview:
            interview.toObject(),
        });
      }

      // ------------------------------------------------------
      // NEXT QUESTION
      // ------------------------------------------------------

      let next;

      // ------------------------------------------------------
      // CROSS QUESTION
      // ------------------------------------------------------

      // ------------------------------------------------------
      // NEXT QUESTION SELECTION (PREVENTS REPEATING QUESTION LOOPS)
      // ------------------------------------------------------
      // Count main (non-cross) questions answered so far
      const mainQuestionsCount = interview.answers.filter(
        a => a.questionType !== 'cross_question' && !a.crossQuestion
      ).length;

      // Check if previous question was ALREADY a cross-question
      const lastAns = interview.answers[interview.answers.length - 1];
      const lastWasCross = Boolean(
        lastAns && (
          lastAns.questionType === 'cross_question' || 
          lastAns.crossQuestion || 
          /cross-question|follow-up/i.test(lastAns.category || '')
        )
      );

      // Check if candidate explicitly opted out or said "I don't know"
      const candidateOptedOut = /i don['’]?t know|no idea|idk|pass|don['’]?t know/i.test(answer);

      // Allow at most ONE follow-up per main question, blocked if candidate opted out or last was cross
      const allowFollowUp = Boolean(
        followUp && 
        !lastWasCross && 
        !candidateOptedOut && 
        (result?.followUpQuestion || evaluation?.followUpQuestion)
      );

      if (allowFollowUp) {
        const currentTier = (Array.isArray(interview.questionBank) && interview.questionBank[mainQuestionsCount - 1]?.difficultyTier) || 'easy';
        next = {
          section: 'verbal',
          questionType: 'cross_question',
          difficultyTier: currentTier,
          text: String(result.followUpQuestion || evaluation.followUpQuestion),
          category: `${currentTier.charAt(0).toUpperCase() + currentTier.slice(1)} / Follow-up`,
          crossQuestion: true,
        };
        console.log(`🎯 Triggered single follow-up for Question ${mainQuestionsCount}`);
      } else {
        // Dynamic question generation: preloaded setup/resume context is reused,
        // while the actual question is based on the candidate's latest answers.
        const classification = classifyInterview(interview.setup);
        let desiredSection = 'verbal';
        if (classification.technical && mainQuestionsCount >= 2) {
          const codingDone = interview.answers.filter(a => a.section === 'coding').length;
          const sqlDone = interview.answers.filter(a => a.section === 'sql').length;
          if (classification.wantsCoding && codingDone < plan.coding && classification.wantsSql && sqlDone < plan.sql) {
            desiredSection = codingDone <= sqlDone ? 'coding' : 'sql';
          } else if (classification.wantsCoding && codingDone < plan.coding) desiredSection = 'coding';
          else if (classification.wantsSql && sqlDone < plan.sql) desiredSection = 'sql';
        }
        try {
          const dynamic = await callWithRetry(() => llm.generateNextQuestion({
            setup: interview.setup, history: historyFrom(interview),
            questionIndex: mainQuestionsCount + 1, section: desiredSection,
            lastEvaluation: evaluation
          }), { maxAttempts: 3, delayMs: 500 });
          next = dynamic?.question || dynamic;
        } catch (generationError) {
          console.warn('Dynamic question generation failed; using local fallback:', generationError?.message || generationError);
          const localBank = generateLocalQuestionBank(interview.setup, plan);
          next = desiredSection === 'verbal'
            ? (localBank.find(q => (q.section || q.type) === 'verbal') || localBank[mainQuestionsCount % localBank.length])
            : (localBank.find(q => (q.section || q.type) === desiredSection) || {
                section: desiredSection, type: desiredSection,
                questionType: desiredSection === 'sql' ? 'sql_workspace_problem' : 'coding_workspace_problem',
                category: desiredSection === 'sql' ? 'SQL / Database' : 'Coding / DSA',
                text: desiredSection === 'sql'
                  ? 'Write a SQL query using the tables and requirements you would expect for this role. Explain your assumptions and edge cases.'
                  : `Solve a practical ${interview.setup.codingLanguage || 'programming'} problem relevant to ${interview.setup.role || interview.setup.subDomain || 'software engineering'} and explain your approach and complexity.`,
                problem: null, constraints: [], examples: [], language: desiredSection === 'sql' ? 'sql' : (interview.setup.codingLanguage || 'python')
              });
        }
      }

      // ------------------------------------------------------
      // SAVE INTERVIEW
      // ------------------------------------------------------

      await interview.save();

      // ------------------------------------------------------
      // RESPONSE
      // ------------------------------------------------------

      return res.json({
        done: false,

        evaluation,

        score,

        followUp,

        followUpReason,

        nextQuestion:
          next,
      });
    } catch (e) {
      console.error(
        'Answer processing failed:',
        e
      );

      return res
        .status(500)
        .json({
          message:
            e?.message ||
            'Failed to process answer',
        });
    }
  };

// ============================================================
// TRANSCRIBE ANSWER AUDIO
// ============================================================

exports.transcribe =
  async (req, res) => {
    try {
      if (!req.file) {
        return res
          .status(400)
          .json({
            message:
              'Audio file is required.',
          });
      }

      // Audio is only passed to Whisper.
      // It is not saved as an interview recording.

      const result =
        await llm.transcribe(
          req.file.buffer,
          req.file.originalname,
          req.file.mimetype
        );

      return res.json(
        result
      );
    } catch (e) {
      console.error(
        'Transcription failed:',
        e
      );

      return res
        .status(500)
        .json({
          message:
            e?.message ||
            'Transcription failed',
        });
    }
  };

// ============================================================
// COMPLETE INTERVIEW MANUALLY
// ============================================================

exports.complete =
  async (req, res) => {
    try {
      const interview =
        await Interview.findOne({
          _id:
            req.params.id,

          userId:
            req.userId,
        });

      if (!interview) {
        return res
          .status(404)
          .json({
            message:
              'Interview not found.',
          });
      }

      if (
        interview.status ===
        'active'
      ) {
        interview.status =
          'completed';

        interview.completedAt =
          new Date();

        interview.durationSeconds =
          interview.startedAt
            ? Math.round(
                (
                  Date.now() -
                  interview.startedAt.getTime()
                ) / 1000
              )
            : 0;

        interview.finalScore =
          averageScore(
            interview.answers
          );

        const lastAnswer = interview.answers[interview.answers.length - 1];
        const lastSummary =
          lastAnswer?.evaluation?.feedback ||
          lastAnswer?.evaluation?.strengths ||
          (interview.answers.length ? `Candidate completed ${interview.answers.length} question(s).` : 'Interview ended by candidate.');

        const isTabViolation = req.body?.reason === 'tab_switching_violation';
        interview.summary = {
          averageScore:
            interview.finalScore,

          questionCount:
            interview.answers.length,

          completedBecause:
            req.body?.reason || 'candidate_ended_interview',

          lastSummary:
            isTabViolation
              ? 'Interview terminated automatically due to repeated tab switching violations.'
              : lastSummary,
        };

        await interview.save();
      }

      return res.json({
        done: true,

        interview:
          interview.toObject(),
      });
    } catch (e) {
      console.error(
        'Complete interview failed:',
        e
      );

      return res
        .status(500)
        .json({
          message:
            e?.message ||
            'Failed to complete interview',
        });
    }
  };

// ============================================================
// LIST COMPLETED INTERVIEWS / REPORTS
// ============================================================

exports.list =
  async (req, res) => {
    try {
      const interviews =
        await Interview.find({
          userId:
            req.userId,

          status:
            'completed',
        })
          .sort({
            completedAt: -1,
            createdAt: -1,
          })
          .lean();

      return res.json({
        interviews,
      });
    } catch (e) {
      console.error(
        'Load reports failed:',
        e
      );

      return res
        .status(500)
        .json({
          message:
            e?.message ||
            'Failed to load reports',
        });
    }
  };

// ============================================================
// GET SINGLE INTERVIEW
// ============================================================

exports.get =
  async (req, res) => {
    try {
      const interview =
        await Interview.findOne({
          _id:
            req.params.id,

          userId:
            req.userId,
        }).lean();

      if (!interview) {
        return res
          .status(404)
          .json({
            message:
              'Interview not found.',
          });
      }

      return res.json(
        interview
      );
    } catch (e) {
      console.error(
        'Get interview failed:',
        e
      );

      return res
        .status(500)
        .json({
          message:
            e?.message ||
            'Failed to get interview',
        });
    }
  };

// ============================================================
// EXPORT HELPERS
// ============================================================

module.exports.cleanSetup =
  cleanSetup;

module.exports.buildPlan =
  buildPlan;

module.exports.callWithRetry =
  callWithRetry;