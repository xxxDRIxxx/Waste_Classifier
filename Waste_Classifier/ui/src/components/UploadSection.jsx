export default function UploadSection() {
  return (
    <div className="max-w-xl mx-auto mt-16 bg-white/10 p-8 rounded-2xl border border-white/20 backdrop-blur-md text-center">
      <h2 className="text-2xl font-semibold mb-4 text-green-400">Upload an Image</h2>
      <input
        type="file"
        className="file:mr-4 file:py-2 file:px-4 file:rounded-full
                   file:border-0 file:text-sm file:font-semibold
                   file:bg-green-500 file:text-white hover:file:bg-green-600"
      />
      <p className="text-sm text-gray-400 mt-3">or use webcam for real-time detection</p>
    </div>
  );
}