// Load variables from .env
require('dotenv').config();

// Now you can access them like this:
const apiKey = process.env.API_KEY;

console.log("Your API Key is:", apiKey);
