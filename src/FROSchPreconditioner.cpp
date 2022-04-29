#ifndef FVM_FROSCH_PRECONDITIONER_CPP
#define FVM_FROSCH_PRECONDITIONER_CPP

namespace FROSch {

    IfpackPreconditioner::IfpackPreconditioner(Teuchos::RCP<const Epetra_RowMatrix> matrix,
                                               Teuchos::RCP<Teuchos::ParameterList> &parameterList):
        Comm_ (Teuchos::rcp(&(matrix->Comm()))),
        Matrix_ (matrix),
        ParameterList_ (parameterList)
    {
        using namespace Teuchos;
        using namespace Xpetra;
        using Xpetra::Matrix;

        const Epetra_MpiComm& tmpComm = dynamic_cast<const Epetra_MpiComm&> (*Comm_);
        TeuchosComm_ = rcp(new MpiComm<int> (tmpComm.Comm()));

        // Convert the matrix
        RCP<const Epetra_CrsMatrix> crsMatrix = rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix);
        RCP<Epetra_CrsMatrix> crsMatrix_nonconst = rcp_const_cast<Epetra_CrsMatrix>(crsMatrix);
        RCP<CrsMatrix<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> > xCrsMatrix = rcp(new EpetraCrsMatrixT<FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>(crsMatrix_nonconst));
        // this is a wrapper to turn the object into an Xpetra object
        RCP<const Matrix<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> > xMatrix = rcp(new CrsMatrixWrap<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>(xCrsMatrix));
        // this is an Xpetra::Matrix that allows 'viewing' the matrix like a block matrix, for instance

        FROSchPreconditioner_.reset(new TwoLevelBlockPreconditioner<double,int>(xMatrix,ParameterList_));
        //ConstXMapPtr repeatedMap = extractRepeatedMap(comm,underlyingLib);
    }

    int IfpackPreconditioner::InitializeNew(Teuchos::RCP<Epetra_Map> repeated_velocity_map,
                                         Teuchos::RCP<Epetra_Map> u_map,
                                         Teuchos::RCP<Epetra_Map> v_map,
                                         Teuchos::RCP<Epetra_Map> w_map,
                                         Teuchos::RCP<Epetra_Map> p_map)
    {
        using namespace Teuchos;
        using namespace Xpetra;

        unsigned dimension = ParameterList_->get("Dimension",3);
        FROSCH_ASSERT(dimension==2||dimension==3,"dimension is neither 2 nor 3.");

        ArrayRCP<unsigned> dofsPerNodeVector = ParameterList_->get("DofsPerNodeVector",Teuchos::null);
        if (dofsPerNodeVector.is_null()) {
            dofsPerNodeVector = ArrayRCP<unsigned>(2);
            if (dimension==2) {
                dofsPerNodeVector[0] = 2;
            } else if (dimension==3) {
                dofsPerNodeVector[0] = 3;
            }
            dofsPerNodeVector[1] = 1;
        }

        ArrayRCP<FROSch::DofOrdering> dofOrderings = ParameterList_->get("DofOrderings",Teuchos::null);
        if (dofOrderings.is_null()) {
            dofOrderings = ArrayRCP<FROSch::DofOrdering>(2);
            dofOrderings[0] = FROSch::NodeWise;
            dofOrderings[1] = FROSch::NodeWise;
        }

        unsigned overlap = ParameterList_->get("Overlap",1);

        // ArrayRCP<RCP<const Epetra_Map> > repeatedMaps = ParameterList_->get("RepeatedMaps",Teuchos::null);
        // FROSCH_ASSERT(!repeatedMaps.is_null(),"repeatedMaps.is_null()");
        // ArrayRCP<RCP<const Map<int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> > > repeatedMapsX(repeatedMaps.size());
        // for (size_t i = 0; i < repeatedMaps.size(); i++) {
        //     FROSCH_ASSERT(!repeatedMaps[i].is_null(),"repeatedMaps[i].is_null()");
        //     repeatedMapsX[i] = FROSch::ConvertToXpetra<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>::ConvertMap( Xpetra::UseEpetra, *repeatedMaps[i], TeuchosComm_ );
        // }

        // repeated_velocity_map->Print(std::cout);
        // p_map->Print(std::cout);

        // RCP<const Epetra_Map> repeatedMap_velocity = ParameterList_->get("repeated_velocity_map",Teuchos::null);
        // RCP<const Epetra_Map> repeatedMap_pressure = ParameterList_->get("p_map",Teuchos::null);
        FROSCH_ASSERT(!repeated_velocity_map.is_null(),"repeatedMap_velocity.is_null()");
        // FROSCH_ASSERT(!repeatedMap_pressure.is_null(),"repeatedMap_pressure.is_null()");
        FROSCH_ASSERT(!p_map.is_null(),"p_map.is_null()");
        ArrayRCP<RCP<const Map<int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> > > repeatedMaps(2);
        // std::cout << repeatedMaps.size() << std::endl;
        // repeatedMaps[0] = FROSch::ConvertToXpetra<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>::ConvertMap( Xpetra::UseEpetra, *repeated_velocity_map, TeuchosComm_ );
        repeatedMaps[0] = FROSch::ConvertToXpetra<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>::ConvertMap( Xpetra::UseEpetra, *repeated_velocity_map, TeuchosComm_ );
        repeatedMaps[1] = FROSch::ConvertToXpetra<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>::ConvertMap( Xpetra::UseEpetra, *p_map, TeuchosComm_ );
        // repeatedMaps[1] = FROSch::ConvertToXpetra<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>::ConvertMap( Xpetra::UseEpetra, *p_map, TeuchosComm_ );

        // ArrayRCP<ArrayRCP<RCP<const Epetra_Map> > > dofMaps = ParameterList_->get("DofMaps",Teuchos::null);
        // FROSCH_ASSERT(!dofMaps.is_null(),"dofMaps.is_null()");
        // ArrayRCP<ArrayRCP<RCP<const Map<int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> > > > dofMapsX(dofMaps.size());
        // for (size_t i = 0; i < dofMaps.size(); i++) {
        //     FROSCH_ASSERT(!dofMaps[i].is_null(),"dofMaps[i].is_null()");
        //     ArrayRCP<RCP<const Map<int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> > > tmpMaps(dofMaps[i].size());
        //     for (size_t j = 0; j < dofMaps[i].size(); j++) {
        //         FROSCH_ASSERT(!dofMaps[i][j].is_null(),"dofMaps[i][j].is_null()");
        //         tmpMaps[j] = FROSch::ConvertToXpetra<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>::ConvertMap( Xpetra::UseEpetra, *dofMaps[i][j], TeuchosComm_ );
        //     }
        //     dofMapsX[i] = tmpMaps;
        // }

        ArrayRCP<ArrayRCP<RCP<const Map<int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> > > > dofMaps(2);
        // std::cout << dofMaps.size() << " " << dimension << std::endl;
        if (dimension==2) {
            ArrayRCP<RCP<const Map<int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> > > velocityMaps(2);
            // RCP<const Epetra_Map> u_map = ParameterList_->get("u_map",Teuchos::null);
            // RCP<const Epetra_Map> v_map = ParameterList_->get("v_map",Teuchos::null);
            FROSCH_ASSERT(!u_map.is_null(),"u_map.is_null()");
            FROSCH_ASSERT(!v_map.is_null(),"v_map.is_null()");
            velocityMaps[0] = FROSch::ConvertToXpetra<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>::ConvertMap( Xpetra::UseEpetra, *u_map, TeuchosComm_ );
            velocityMaps[1] = FROSch::ConvertToXpetra<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>::ConvertMap( Xpetra::UseEpetra, *v_map, TeuchosComm_ );
            dofMaps[0] = velocityMaps;
        } else if (dimension==3) {
            ArrayRCP<RCP<const Map<int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> > > velocityMaps(3);
            // RCP<const Epetra_Map> u_map = ParameterList_->get("u_map",Teuchos::null);
            // RCP<const Epetra_Map> v_map = ParameterList_->get("v_map",Teuchos::null);
            // RCP<const Epetra_Map> w_map = ParameterList_->get("w_map",Teuchos::null);
            FROSCH_ASSERT(!u_map.is_null(),"u_map.is_null()");
            FROSCH_ASSERT(!v_map.is_null(),"v_map.is_null()");
            FROSCH_ASSERT(!w_map.is_null(),"w_map.is_null()");
            velocityMaps[0] = FROSch::ConvertToXpetra<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>::ConvertMap( Xpetra::UseEpetra, *u_map, TeuchosComm_ );
            velocityMaps[1] = FROSch::ConvertToXpetra<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>::ConvertMap( Xpetra::UseEpetra, *v_map, TeuchosComm_ );
            velocityMaps[2] = FROSch::ConvertToXpetra<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>::ConvertMap( Xpetra::UseEpetra, *w_map, TeuchosComm_ );
            dofMaps[0] = velocityMaps;
        }
        ArrayRCP<RCP<const Map<int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> > > pressureMaps(1);
        // RCP<const Epetra_Map> p_map = ParameterList_->get("p_map",Teuchos::null);
        FROSCH_ASSERT(!p_map.is_null(),"p_map.is_null()");
        pressureMaps[0] = FROSch::ConvertToXpetra<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>::ConvertMap( Xpetra::UseEpetra, *p_map, TeuchosComm_ );
        dofMaps[1] = pressureMaps;

        FROSchPreconditioner_->initialize(dimension,
                                          dofsPerNodeVector,
                                          dofOrderings,
                                          overlap,
                                          repeatedMaps,
                                          null,
                                          null,
                                          dofMaps);

        IsInitialized_ = true;
        return 0;
    }

    int IfpackPreconditioner::Compute()
    {
        FROSCH_ASSERT(IsInitialized_==true,"IsInitialized_==false.");
        FROSchPreconditioner_->compute();
        IsComputed_ = true;
        return 0;
    }

    int IfpackPreconditioner::ApplyInverse(const Epetra_MultiVector &X,
                                           Epetra_MultiVector &Y) const
    {
        using namespace Teuchos;
        using namespace Xpetra;

        FROSCH_ASSERT(IsComputed_==true,"IsInitialized_==false.");

        RCP<const MultiVector<double,int,FROSch::DefaultGlobalOrdinal> > xX = rcp(new EpetraMultiVectorT<FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>(rcpFromRef(const_cast<Epetra_MultiVector&>(X))));
        RCP<MultiVector<double,int,FROSch::DefaultGlobalOrdinal> > xY = rcp(new EpetraMultiVectorT<FROSch::DefaultGlobalOrdinal, KokkosClassic::DefaultNode::DefaultNodeType>(rcpFromRef(Y)));

        FROSchPreconditioner_->apply(*xX,*xY);
        return 0;
    }

} // namespace FROSch

#endif
