// PFNanoLite: a minimal, self-contained CMSSW EDAnalyzer that flattens the
// MiniAOD muons + ALL particle-flow candidates (+ lostTracks) into a flat ROOT
// TTree named "Events", using the SAME branch names as the analysis pipeline
// (Muon_*, PFCands_*). It reads only already-reconstructed MiniAOD collections,
// so it needs NO global-tag / Frontier conditions -- which keeps the portable
// one-command open-data workflow simple and robust.
//
// This is the "_allPF" content the published jets-only PFNano (record 31305)
// omits: every packedPFCandidate, not just jet constituents.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "TTree.h"
#include <vector>

class PFNanoLite : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit PFNanoLite(const edm::ParameterSet&);
  ~PFNanoLite() override = default;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void addCands(const std::vector<pat::PackedCandidate>&);

  edm::EDGetTokenT<std::vector<pat::Muon>> muTok_;
  edm::EDGetTokenT<std::vector<pat::PackedCandidate>> pfTok_;
  edm::EDGetTokenT<std::vector<pat::PackedCandidate>> lostTok_;
  bool useLost_;
  double pfPtMin_;

  TTree* tree_;
  // muons
  unsigned nMuon_;
  std::vector<float> Muon_pt_, Muon_eta_, Muon_phi_, Muon_mass_, Muon_charge_;
  // pf candidates
  unsigned nPFCands_;
  std::vector<float> PFCands_pt_, PFCands_eta_, PFCands_phi_, PFCands_mass_;
  std::vector<float> PFCands_charge_, PFCands_puppiWeight_, PFCands_dz_;
  std::vector<int> PFCands_pdgId_, PFCands_pvAssocQuality_;
};

PFNanoLite::PFNanoLite(const edm::ParameterSet& ps)
    : muTok_(consumes(ps.getParameter<edm::InputTag>("muons"))),
      pfTok_(consumes(ps.getParameter<edm::InputTag>("pfCandidates"))),
      lostTok_(consumes(ps.getParameter<edm::InputTag>("lostTracks"))),
      useLost_(ps.getParameter<bool>("useLostTracks")),
      pfPtMin_(ps.getParameter<double>("pfPtMin")) {
  usesResource("TFileService");
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("Events", "Events");
  tree_->Branch("nMuon", &nMuon_);
  tree_->Branch("Muon_pt", &Muon_pt_);
  tree_->Branch("Muon_eta", &Muon_eta_);
  tree_->Branch("Muon_phi", &Muon_phi_);
  tree_->Branch("Muon_mass", &Muon_mass_);
  tree_->Branch("Muon_charge", &Muon_charge_);
  tree_->Branch("nPFCands", &nPFCands_);
  tree_->Branch("PFCands_pt", &PFCands_pt_);
  tree_->Branch("PFCands_eta", &PFCands_eta_);
  tree_->Branch("PFCands_phi", &PFCands_phi_);
  tree_->Branch("PFCands_mass", &PFCands_mass_);
  tree_->Branch("PFCands_charge", &PFCands_charge_);
  tree_->Branch("PFCands_pdgId", &PFCands_pdgId_);
  tree_->Branch("PFCands_puppiWeight", &PFCands_puppiWeight_);
  tree_->Branch("PFCands_dz", &PFCands_dz_);
  tree_->Branch("PFCands_pvAssocQuality", &PFCands_pvAssocQuality_);
}

void PFNanoLite::addCands(const std::vector<pat::PackedCandidate>& cands) {
  for (const auto& c : cands) {
    if (c.pt() < pfPtMin_) continue;
    PFCands_pt_.push_back(c.pt());
    PFCands_eta_.push_back(c.eta());
    PFCands_phi_.push_back(c.phi());
    PFCands_mass_.push_back(c.mass());
    PFCands_charge_.push_back(c.charge());
    PFCands_pdgId_.push_back(c.pdgId());
    PFCands_puppiWeight_.push_back(c.puppiWeight());
    PFCands_dz_.push_back(c.charge() != 0 ? c.dz() : 0.f);
    PFCands_pvAssocQuality_.push_back(c.pvAssociationQuality());
  }
}

void PFNanoLite::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  Muon_pt_.clear(); Muon_eta_.clear(); Muon_phi_.clear();
  Muon_mass_.clear(); Muon_charge_.clear();
  PFCands_pt_.clear(); PFCands_eta_.clear(); PFCands_phi_.clear();
  PFCands_mass_.clear(); PFCands_charge_.clear(); PFCands_pdgId_.clear();
  PFCands_puppiWeight_.clear(); PFCands_dz_.clear(); PFCands_pvAssocQuality_.clear();

  const auto& muons = iEvent.get(muTok_);
  for (const auto& m : muons) {
    Muon_pt_.push_back(m.pt());
    Muon_eta_.push_back(m.eta());
    Muon_phi_.push_back(m.phi());
    Muon_mass_.push_back(m.mass());
    Muon_charge_.push_back(m.charge());
  }
  nMuon_ = Muon_pt_.size();

  addCands(iEvent.get(pfTok_));
  if (useLost_) addCands(iEvent.get(lostTok_));
  nPFCands_ = PFCands_pt_.size();

  tree_->Fill();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFNanoLite);
