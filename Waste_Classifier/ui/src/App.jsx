import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import UploadSection from "./components/UploadSection";
import Footer from "./components/Footer";

function App() {
  return (
    <div className="bg-gradient-to-b from-slate-950 via-slate-900 to-slate-800 min-h-screen text-white">
      <Navbar />
      <Hero />
      <UploadSection />
      <Footer />
    </div>
  );
}

export default App;