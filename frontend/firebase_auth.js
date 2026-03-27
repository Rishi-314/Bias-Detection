// ═══════════════════════════════════════════════════════════════════
//  firebase-auth.js — FairTrust AI · Firebase Authentication Helper
//
//  HOW TO SET UP FIREBASE (do this first!):
//  1. Go to https://console.firebase.google.com
//  2. Click "Add project" → name it "fairtrust-ai" → Continue
//  3. Disable Google Analytics (optional) → Create project
//  4. Click the </> (Web) icon to register a web app
//  5. Give it a nickname (e.g. "FairTrust Web") → Register app
//  6. Copy the firebaseConfig object below and REPLACE the placeholder
//     values with your actual config from Firebase Console
//  7. In Firebase Console → Authentication → Get started → Sign-in method
//     → Enable "Email/Password" and optionally "Google"
//  8. Done! Your users can now sign up and log in.
// ═══════════════════════════════════════════════════════════════════

// ── PASTE YOUR FIREBASE CONFIG HERE ──────────────────────────────────
// Replace ALL placeholder values with your actual Firebase project config
const FIREBASE_CONFIG = {
  apiKey:            "YOUR_API_KEY",
  authDomain:        "YOUR_PROJECT_ID.firebaseapp.com",
  projectId:         "YOUR_PROJECT_ID",
  storageBucket:     "YOUR_PROJECT_ID.appspot.com",
  messagingSenderId: "YOUR_SENDER_ID",
  appId:             "YOUR_APP_ID",
};
// ─────────────────────────────────────────────────────────────────────

// Import Firebase from CDN (compatible with plain HTML files)
import { initializeApp }                                    from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getAuth, createUserWithEmailAndPassword,
         signInWithEmailAndPassword, signOut,
         onAuthStateChanged, GoogleAuthProvider,
         signInWithPopup, updateProfile,
         sendPasswordResetEmail }                           from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";

// Initialise
const app  = initializeApp(FIREBASE_CONFIG);
const auth = getAuth(app);

// ── User-friendly error messages ──────────────────────────────────────
function friendlyError(code) {
  const map = {
    "auth/email-already-in-use":   "An account with this email already exists.",
    "auth/invalid-email":          "Please enter a valid email address.",
    "auth/weak-password":          "Password must be at least 6 characters.",
    "auth/user-not-found":         "No account found with this email.",
    "auth/wrong-password":         "Incorrect password. Please try again.",
    "auth/too-many-requests":      "Too many attempts. Please try again in a few minutes.",
    "auth/network-request-failed": "Network error. Check your internet connection.",
    "auth/popup-closed-by-user":   "Google sign-in was cancelled.",
    "auth/invalid-credential":     "Email or password is incorrect.",
  };
  return map[code] || "Something went wrong. Please try again.";
}

// ── Auth state listener — call this on every protected page ──────────
// Usage: onAuth(user => { if (!user) redirect to login })
function onAuth(callback) {
  return onAuthStateChanged(auth, callback);
}

// ── Sign Up ───────────────────────────────────────────────────────────
async function signUp(name, email, password) {
  const cred = await createUserWithEmailAndPassword(auth, email, password);
  await updateProfile(cred.user, { displayName: name });
  return cred.user;
}

// ── Sign In ───────────────────────────────────────────────────────────
async function signIn(email, password) {
  const cred = await signInWithEmailAndPassword(auth, email, password);
  return cred.user;
}

// ── Google Sign-In ────────────────────────────────────────────────────
async function signInWithGoogle() {
  const provider = new GoogleAuthProvider();
  const cred     = await signInWithPopup(auth, provider);
  return cred.user;
}

// ── Sign Out ──────────────────────────────────────────────────────────
async function logOut() {
  await signOut(auth);
  window.location.href = "login.html";
}

// ── Password Reset ────────────────────────────────────────────────────
async function resetPassword(email) {
  await sendPasswordResetEmail(auth, email);
}

// ── Get current user ──────────────────────────────────────────────────
function currentUser() {
  return auth.currentUser;
}

export {
  auth, onAuth, signUp, signIn, signInWithGoogle,
  logOut, resetPassword, currentUser, friendlyError
};