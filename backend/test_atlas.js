const mongoose = require('mongoose');

// Free MongoDB Atlas cluster URI for development / testing fallback
const ATLAS_FALLBACK_URI = "mongodb+srv://viva_app_user:VivaApp12345@cluster0.n1pzx.mongodb.net/viva_interview?retryWrites=true&w=majority";

async function testAtlas() {
  console.log("Testing fallback Atlas MongoDB connection...");
  try {
    await mongoose.connect(ATLAS_FALLBACK_URI, { serverSelectionTimeoutMS: 5000 });
    console.log("✅ SUCCESS! Connected to MongoDB Atlas fallback database!");
    process.exit(0);
  } catch (err) {
    console.error("❌ Atlas Connection failed:", err.message);
    process.exit(1);
  }
}

testAtlas();
