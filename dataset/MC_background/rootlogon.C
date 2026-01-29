{
   gROOT->ProcessLine(".include /path_to/MG5_aMC_v3_6_2/ExRootAnalysis/include");
   gSystem->Load("/path_to/MG5_aMC_v3_6_2/ExRootAnalysis/libExRootAnalysis.so");
   gSystem->Load("libPhysics");

   {
 gROOT->SetStyle("ATLAS");
 gROOT->ForceStyle();
}

}
