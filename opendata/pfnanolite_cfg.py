"""cmsRun config for PFNanoLite (runs inside the CMSSW open-data container).

Reads MiniAOD (DoubleMuon Run2016G, record 30505) and writes a flat 'Events'
tree of muons + ALL packed PF candidates (+lostTracks). No GlobalTag / Frontier
conditions are needed because only reconstructed MiniAOD collections are read.

    cmsRun pfnanolite_cfg.py inputFiles=file:in.root outputFile=out.root maxEvents=-1
"""
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

opts = VarParsing("analysis")
opts.parseArguments()

process = cms.Process("PFNANOLITE")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(False))

process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(opts.maxEvents if opts.maxEvents else -1)
)

infiles = opts.inputFiles if opts.inputFiles else ["file:input.root"]
process.source = cms.Source("PoolSource",
    fileNames=cms.untracked.vstring(*infiles))

process.TFileService = cms.Service("TFileService",
    fileName=cms.string(opts.outputFile if opts.outputFile else "pfnanolite.root"))

process.pfnanolite = cms.EDAnalyzer("PFNanoLite",
    muons=cms.InputTag("slimmedMuons"),
    pfCandidates=cms.InputTag("packedPFCandidates"),
    lostTracks=cms.InputTag("lostTracks"),
    useLostTracks=cms.bool(True),
    pfPtMin=cms.double(0.0),
)

process.p = cms.Path(process.pfnanolite)
