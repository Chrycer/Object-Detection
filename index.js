// Import the functions you need from the SDKs you need
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.3.1/firebase-app.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/11.3.1/firebase-analytics.js";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
apiKey: "AIzaSyCJuwmPGXDDzE3XTp3UAhkz9b9NrHT5rVE",
authDomain: "accenture-a5bf3.firebaseapp.com",
projectId: "accenture-a5bf3",
storageBucket: "accenture-a5bf3.firebasestorage.app",
messagingSenderId: "371975684408",
appId: "1:371975684408:web:a36df4ab44df836cfa81cd",
measurementId: "G-2QJNWDPKTN"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);