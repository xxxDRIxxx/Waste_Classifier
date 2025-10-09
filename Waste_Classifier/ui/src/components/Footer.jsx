export default function Footer() {
  return (
    <footer className="mt-20 py-6 text-center text-gray-500 border-t border-white/10">
      <p>© {new Date().getFullYear()} Waste Classifier. All rights reserved.</p>
    </footer>
  );
}