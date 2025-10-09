import { motion } from "framer-motion";

export default function Hero() {
  return (
    <section className="flex flex-col items-center justify-center py-24 text-center px-6">
      <motion.h1 
        className="text-5xl font-extrabold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-green-300 to-green-500"
        initial={{ opacity: 0, y: -40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        AI-Powered Waste Classification
      </motion.h1>
      <p className="max-w-xl text-gray-300 text-lg mb-10">
        Upload or capture an image and let our model detect and classify waste accurately.
      </p>
      <motion.button
        className="px-6 py-3 bg-green-500 hover:bg-green-600 rounded-xl text-white font-semibold shadow-lg"
        whileHover={{ scale: 1.05 }}
      >
        Start Classifying ♻
      </motion.button>
    </section>
  );
}