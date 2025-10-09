import React from 'react'

export default function App() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-green-200 via-white to-green-300 text-gray-800">
      <div className="bg-white p-8 rounded-2xl shadow-2xl max-w-lg w-full text-center">
        <h1 className="text-4xl font-extrabold mb-4 text-green-600">♻️ Waste Classifier</h1>
        <p className="mb-6 text-gray-600">Welcome to the intelligent waste classification system.</p>
        <div className="grid grid-cols-2 gap-4">
          <button className="px-4 py-3 rounded-xl bg-green-500 text-white font-semibold hover:bg-green-600 transition">📸 Webcam</button>
          <button className="px-4 py-3 rounded-xl bg-blue-500 text-white font-semibold hover:bg-blue-600 transition">📂 Upload</button>
        </div>
        <p className="mt-6 text-sm text-gray-500">Powered by YOLOv8 + Streamlit + React</p>
      </div>
    </div>
  )
}
