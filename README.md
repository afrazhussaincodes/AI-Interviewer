# VIVA AI Interview

## Run
1. Start MongoDB.
2. `cd backend && npm install`
3. Copy `.env.example` to `.env` and adjust values.
4. Install Python dependencies: `pip install -r requirements.txt`.
5. From `backend`, run `npm start`. This starts the Node API on port 4000 and the Python proctor on port 5000.
6. In the project root, run `npm install` then `npm run dev`.

The frontend talks to `http://localhost:4000/api`. The interview room talks to the Python proctor at `http://localhost:5000`.

## Authentication/profile flow
- Sign up creates a MongoDB user and signs them in.
- Signup redirects to Profile.
- A profile is not considered complete until all required profile fields AND a parsed resume are present.
- Dashboard/interview tools are guarded until completion.
- Resume upload parses PDF/DOCX/TXT, stores the text and file metadata in MongoDB, and autofills editable profile fields.
- Interview Setup reads the stored profile/resume every time it opens.
- Resume Analysis calls the backend and displays the live ATS score.
