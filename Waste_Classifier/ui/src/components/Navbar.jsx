export default function Navbar() {
  return (
    <nav className="flex justify-between items-center px-8 py-4 bg-white/5 backdrop-blur-lg sticky top-0 z-50">
      <h1 className="text-2xl font-bold text-green-400">♻ Waste Classifier</h1>
      <ul className="flex gap-6 text-lg">
        <li className="hover:text-green-300 cursor-pointer">Home</li>
        <li className="hover:text-green-300 cursor-pointer">About</li>
        <li className="hover:text-green-300 cursor-pointer">Contact</li>
      </ul>
    </nav>
  );
}