#include "TChain.h"
#include "TH1F.h"
#include "TLorentzVector.h"
#include "TCanvas.h"
#include "TMath.h"
#include "ExRootAnalysis/ExRootTreeReader.h"
#include "ExRootAnalysis/ExRootClasses.h"
#include <fstream>
#include <iostream>
#include <iomanip>

void processFile(const char* inputFile, const char* outputFile)
{
    // Load the LHEF events file into a TChain
    TChain chain("LHEF");
    chain.Add(inputFile);

    std::string csvFileName = std::string(outputFile) + ".csv";
    std::ofstream csvFile(csvFileName);
    csvFile << std::setprecision(15) << std::scientific;

    // Write the header line with column names
    csvFile << "l0_id,l1_id,q0_id,q1_id,"
            << "l0_pt,l1_pt,q0_pt,q1_pt,"
            << "l0_phi,l1_phi,q0_phi,q1_phi,"
            << "l0_eta,l1_eta,q0_eta,q1_eta,"
            << "l0_m,l1_m,q0_m,q1_m,"
            << "l0_e,l1_e,q0_e,q1_e,"
            << "met_et,met_phi,m_ll,m_qq,"
            << "pt_ll,pt_qq,d_phi_ll,d_phi_qq,"
            << "d_eta_ll,d_eta_qq,d_y_ll,d_y_qq,"
            << "sqrtHT,MET_sig,m_l0q0,m_l0q1,m_l1q0,m_l1q1\n";

    int l0_id, l1_id, q0_id, q1_id;
    double l0_pt, l1_pt, q0_pt, q1_pt;
    double l0_phi, l1_phi, q0_phi, q1_phi;
    double l0_eta, l1_eta, q0_eta, q1_eta;
    double l0_m, l1_m, q0_m, q1_m;
    double l0_e, l1_e, q0_e, q1_e;
    double met_et, met_phi, m_ll, m_qq;
    double pt_ll, pt_qq, d_phi_ll, d_phi_qq;
    double d_eta_ll, d_eta_qq, d_y_ll, d_y_qq;
    double sqrtHT, MET_sig, m_l0q0, m_l0q1, m_l1q0, m_l1q1;

    ExRootTreeReader *treeReader = new ExRootTreeReader(&chain);
    Long64_t numberOfEntries = treeReader->GetEntries();

    TClonesArray *branchParticle = treeReader->UseBranch("Particle");

    for(Long64_t entry = 0; entry < numberOfEntries; ++entry)
    {
        treeReader->ReadEntry(entry);
        if (entry % 100000 == 0) cout << "Processing Event " << entry << endl;

        TLorentzVector electron, positron;
        int electron_id = 0, positron_id = 0;
        TLorentzVector upQuark, downQuark;
        int upQuark_id = 0, downQuark_id = 0;

        // Variables for MET
        double met_px = 0.0;
        double met_py = 0.0;

        // Loop over particles in the event
        for(Int_t part_i = 0; part_i < branchParticle->GetEntries(); ++part_i)
        {
            TRootLHEFParticle *particle = (TRootLHEFParticle*) branchParticle->At(part_i);

            // Electrons and positrons
            if(particle->PID == 11) // Electron
            {
                electron.SetPtEtaPhiE(particle->PT, particle->Eta, particle->Phi, particle->E);
                electron_id = particle->PID;
            }
            else if(particle->PID == -11) // Positron
            {
                positron.SetPtEtaPhiE(particle->PT, particle->Eta, particle->Phi, particle->E);
                positron_id = particle->PID;
            }

            // Up and down quarks
            else if(particle->PID == 2) // Up quark
            {
                upQuark.SetPtEtaPhiE(particle->PT, particle->Eta, particle->Phi, particle->E);
                upQuark_id = particle->PID;
            }
            else if(particle->PID == 1) // Down quark
            {
                downQuark.SetPtEtaPhiE(particle->PT, particle->Eta, particle->Phi, particle->E);
                downQuark_id = particle->PID;
            }

            // Neutrinos contribute to MET
            else if(abs(particle->PID) == 12 || abs(particle->PID) == 14 || abs(particle->PID) == 16) // Neutrinos
            {
                met_px += particle->PT * cos(particle->Phi);
                met_py += particle->PT * sin(particle->Phi);
            }
        }

        // Calculate MET
        met_et = sqrt(met_px * met_px + met_py * met_py);
        met_phi = atan2(met_py, met_px);

        if(electron.Pt() > 0 && positron.Pt() > 0 && upQuark.Pt() > 0 && downQuark.Pt() > 0)
        {
            // Determine the most energetic lepton
            TLorentzVector l0, l1;
            int l0_id_temp, l1_id_temp;

            if(electron.Pt() > positron.Pt())
            {
                l0 = electron;
                l0_id_temp = electron_id;
                l1 = positron;
                l1_id_temp = positron_id;
            }
            else
            {
                l0 = positron;
                l0_id_temp = positron_id;
                l1 = electron;
                l1_id_temp = electron_id;
            }

            // Determine the most energetic quark
            TLorentzVector q0, q1;
            int q0_id_temp, q1_id_temp;

            if(upQuark.Pt() > downQuark.Pt())
            {
                q0 = upQuark;
                q0_id_temp = upQuark_id;
                q1 = downQuark;
                q1_id_temp = downQuark_id;
            }
            else
            {
                q0 = downQuark;
                q0_id_temp = downQuark_id;
                q1 = upQuark;
                q1_id_temp = upQuark_id;
            }

            // Transverse momentum
            l0_pt = l0.Pt();
            l1_pt = l1.Pt();
            q0_pt = q0.Pt();
            q1_pt = q1.Pt();

            // Phi
            l0_phi = l0.Phi();
            l1_phi = l1.Phi();
            q0_phi = q0.Phi();
            q1_phi = q1.Phi();

            // Eta
            l0_eta = l0.Eta();
            l1_eta = l1.Eta();
            q0_eta = q0.Eta();
            q1_eta = q1.Eta();

            // Mass
            l0_m = l0.M();
            l1_m = l1.M();
            q0_m = q0.M();
            q1_m = q1.M();

            // Energy
            l0_e = l0.E();
            l1_e = l1.E();
            q0_e = q0.E();
            q1_e = q1.E();

            // Particle IDs
            l0_id = l0_id_temp;
            l1_id = l1_id_temp;
            q0_id = q0_id_temp;
            q1_id = q1_id_temp;

            // Center-of-mass energies
            m_ll = (l0 + l1).M();
            m_qq = (q0 + q1).M();

            // Transverse momentum of systems
            pt_ll = (l0 + l1).Pt();
            pt_qq = (q0 + q1).Pt();

            // Angle differences
            d_phi_ll = std::fmod((l0.Eta() > l1.Eta()) * (l0.Phi() - l1.Phi()) + (l1.Eta() > l0.Eta()) * (l1.Phi() - l0.Phi()) + 2. * M_PI, 2. * M_PI);
            d_phi_qq = std::fmod((q0.Eta() > q1.Eta()) * (q0.Phi() - q1.Phi()) + (q1.Eta() > q0.Eta()) * (q1.Phi() - q0.Phi()) + 2. * M_PI, 2. * M_PI);

            // Pseudorapidity differences
            d_eta_ll = l0.Eta() - l1.Eta();
            d_eta_qq = q0.Eta() - q1.Eta();

            // Rapidity differences
            d_y_ll = l0.Rapidity() - l1.Rapidity();
            d_y_qq = q0.Rapidity() - q1.Rapidity();

            // sqrtHT
            sqrtHT = sqrt(l0_pt + l1_pt + q0_pt + q1_pt + met_et);

            // MET significance
            MET_sig = met_et / sqrtHT;

            // Center-of-mass energies for lepton-quark pairs
            m_l0q0 = (l0 + q0).M();
            m_l0q1 = (l0 + q1).M();
            m_l1q0 = (l1 + q0).M();
            m_l1q1 = (l1 + q1).M();


            // Write data to CSV file
            csvFile << l0_id << "," << l1_id << "," << q0_id << "," << q1_id << ","
                    << l0_pt << "," << l1_pt << "," << q0_pt << "," << q1_pt << ","
                    << l0_phi << "," << l1_phi << "," << q0_phi << "," << q1_phi << ","
                    << l0_eta << "," << l1_eta << "," << q0_eta << "," << q1_eta << ","
                    << l0_m << "," << l1_m << "," << q0_m << "," << q1_m << ","
                    << l0_e << "," << l1_e << "," << q0_e << "," << q1_e << ","
                    << met_et << "," << met_phi << "," << m_ll << "," << m_qq << ","
                    << pt_ll << "," << pt_qq << "," << d_phi_ll << "," << d_phi_qq << ","
                    << d_eta_ll << "," << d_eta_qq << "," << d_y_ll << "," << d_y_qq << ","
                    << sqrtHT << "," << MET_sig << "," << m_l0q0 << "," << m_l0q1 << ","
                    << m_l1q0 << "," << m_l1q1 << "\n";
        }
    }

    csvFile.close();
}

void ana()
{
    const char* inputFiles[] = {
        // Negative values -10 to -2
        "path_to/raw_root_files/events_cHW-10.000000/unweighted_events.root",
        "path_to/raw_root_files/events_cHW-9.000000/unweighted_events.root",
        "path_to/raw_root_files/events_cHW-8.000000/unweighted_events.root",
        "path_to/raw_root_files/events_cHW-7.000000/unweighted_events.root",
        "path_to/raw_root_files/events_cHW-6.000000/unweighted_events.root",
        "path_to/raw_root_files/events_cHW-5.000000/unweighted_events.root",
        "path_to/raw_root_files/events_cHW-4.000000/unweighted_events.root",
        "path_to/raw_root_files/events_cHW-3.000000/unweighted_events.root",
        "path_to/raw_root_files/events_cHW-2.000000/unweighted_events.root",
        // Positive values 2 to 10
        "path_to/raw_root_files/events_cHW2.000000/unweighted_events.root",
        "path_to/raw_root_files/events_cHW3.000000/unweighted_events.root",
        "path_to/raw_root_files/events_cHW4.000000/unweighted_events.root",
        "path_to/raw_root_files/events_cHW5.000000/unweighted_events.root",
        "path_to/raw_root_files/events_cHW6.000000/unweighted_events.root",
        "path_to/raw_root_files/events_cHW7.000000/unweighted_events.root",
        "path_to/raw_root_files/events_cHW8.000000/unweighted_events.root",
        "path_to/raw_root_files/events_cHW9.000000/unweighted_events.root",
        "path_to/raw_root_files/events_cHW10.000000/unweighted_events.root"
    };

    const char* outputFiles[] = {
        // Negative values -10 to -2
        "output_bsm_-10",
        "output_bsm_-9",
        "output_bsm_-8",
        "output_bsm_-7",
        "output_bsm_-6",
        "output_bsm_-5",
        "output_bsm_-4",
        "output_bsm_-3",
        "output_bsm_-2",
        // Positive values 2 to 10
        "output_bsm_2",
        "output_bsm_3",
        "output_bsm_4",
        "output_bsm_5",
        "output_bsm_6",
        "output_bsm_7",
        "output_bsm_8",
        "output_bsm_9",
        "output_bsm_10"
    };

    for(int i = 0; i < 18; ++i)
    {
        processFile(inputFiles[i], outputFiles[i]);
    }
}